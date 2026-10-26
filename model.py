"""Definisce l'architettura neurale object-centric per generare scene graph."""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    # PyTorch Geometric è opzionale: il backend denso funziona senza questa dipendenza.
    from torch_geometric.nn import GATConv
except ImportError:
    GATConv = None


# Nome ufficiale del checkpoint torchvision usato per la ricerca nella cache.
RESNET34_CHECKPOINT = "resnet34-b627a593.pth"


def _candidate_torch_cache_roots():
    """Ritorna i percorsi candidati usati da PyTorch per i checkpoint dei pesi."""
    # Rispetta lo stesso ordine di precedenza delle cache configurabili di PyTorch.
    roots = []
    torch_home = os.getenv("TORCH_HOME")
    if torch_home:
        roots.append(torch_home)

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        roots.append(os.path.join(xdg_cache_home, "torch"))

    roots.append(os.path.expanduser("~/.cache/torch"))
    return roots


def find_cached_resnet34_weights():
    """Cerca i pesi ImageNet di ResNet34 gia presenti in cache senza scaricare nulla."""
    # Evita accessi di rete: controlla esclusivamente checkpoint locali.
    for root in _candidate_torch_cache_roots():
        checkpoint_path = os.path.join(root, "hub", "checkpoints", RESNET34_CHECKPOINT)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
    return None


def build_resnet34(backbone_init):
    """Costruisce la ResNet34 con una policy robusta per i pesi pretrained."""
    # La modalità random non tenta mai di recuperare pesi esterni.
    if backbone_init == "random":
        return models.resnet34(weights=None), "random"

    # La modalità auto usa ImageNet solo quando il file è già disponibile in cache.
    if backbone_init == "auto":
        cached_weights = find_cached_resnet34_weights()
        model = models.resnet34(weights=None)
        if cached_weights is not None:
            state_dict = torch.load(cached_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            return model, "imagenet_cached"

        warnings.warn(
            "ResNet34 ImageNet weights not found in cache. Falling back to random initialization. "
            "Use --backbone-init imagenet after downloading weights if you want pretrained features.",
            stacklevel=2,
        )
        return model, "random_fallback"

    # La modalità esplicita delega a torchvision anche l'eventuale download.
    if backbone_init == "imagenet":
        try:
            return models.resnet34(weights=models.ResNet34_Weights.DEFAULT), "imagenet"
        except Exception as exc:
            raise RuntimeError(
                "Unable to load ImageNet weights for ResNet34. "
                "If you are offline, use --backbone-init auto or --backbone-init random."
            ) from exc

    # Blocca valori non previsti anche quando il modello è creato fuori dalla CLI.
    raise ValueError(f"Unsupported backbone_init: {backbone_init}")


class CNNBackbone(nn.Module):
    """Backbone visivo basato su ResNet."""

    def __init__(self, backbone_init="auto"):
        """Inizializza ResNet34 senza i livelli finali di pooling e classificazione."""
        super().__init__()
        # Conserva solo l'estrattore convoluzionale fino all'ultima feature map.
        resnet, resolved_init = build_resnet34(backbone_init)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 512
        self.backbone_init = resolved_init

    def forward(self, x):
        """Estrae una mappa di caratteristiche visive dal batch di immagini."""
        return self.backbone(x)


class SoftPositionEmbed(nn.Module):
    """Aggiunge embedding di posizione 2D apprendibili alle mappe delle caratteristiche."""

    def __init__(self, hidden_size, resolution):
        """Crea la proiezione delle coordinate e registra la griglia spaziale."""
        super().__init__()
        # Proietta (x, y, 1-x, 1-y) nella dimensione delle feature.
        self.dense = nn.Linear(4, hidden_size)
        self.register_buffer("grid", self.build_grid(resolution))

    def build_grid(self, resolution):
        """Costruisce coordinate normalizzate dirette e complementari."""
        # Genera assi normalizzati indipendenti per altezza e larghezza.
        ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
        grid = torch.meshgrid(*ranges, indexing='ij')
        grid = torch.stack(grid, dim=-1)
        grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
        grid = grid.unsqueeze(0)
        return torch.cat([grid, 1.0 - grid], dim=-1)

    def forward(self, inputs):
        """Somma alle feature l'embedding appreso della posizione."""
        return inputs + self.dense(self.grid)


class SlotAttention(nn.Module):
    """Modulo Slot Attention per l'aggregazione di token orientata agli oggetti."""

    def __init__(self, num_slots, in_dim, slot_dim, iters=3):
        """Configura parametri degli slot, proiezioni attentive, GRU e MLP."""
        super().__init__()
        # Memorizza dimensioni e fattore di scala dell'attenzione dot-product.
        self.num_slots = num_slots
        self.iters = iters
        self.scale = slot_dim ** -0.5

        # Gli slot iniziali sono campionati da una Gaussiana appresa.
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)

        # Proiezioni query-key-value usate per assegnare token agli slot.
        self.project_q = nn.Linear(slot_dim, slot_dim)
        self.project_k = nn.Linear(in_dim, slot_dim)
        self.project_v = nn.Linear(in_dim, slot_dim)

        # La GRU integra gli aggiornamenti preservando lo stato dello slot.
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )

        # Normalizzazioni separate stabilizzano attenzione e aggiornamento residuo.
        self.norm_input = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        """Raggruppa i token visivi in slot tramite raffinamento iterativo."""
        # Normalizza i token e prepara un campione iniziale per ogni slot del batch.
        B, _, _ = inputs.shape
        inputs = self.norm_input(inputs)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = torch.exp(self.slots_logsigma)
        slots = mu + sigma * torch.randn_like(mu)

        # Key e value dipendono solo dagli input e sono calcolati una volta.
        keys = self.project_k(inputs)
        values = self.project_v(inputs)

        # Raffina ripetutamente gli slot usando attenzione competitiva sui token.
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            queries = self.project_q(slots_norm)

            # Calcola l'affinità slot-token in float32 per maggiore stabilità con AMP.
            attn_logits = torch.einsum('bid,bjd->bij', queries.float(), keys.float()) * self.scale
            attn = F.softmax(attn_logits, dim=-1).to(values.dtype)
            # Aggiunge un piccolo epsilon per la stabilità numerica
            attn = attn + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Aggrega i value dei token secondo i pesi assegnati a ogni slot.
            updates = torch.einsum('bij,bjd->bid', attn, values)

            # Aggiorna gli slot e applica una MLP residua dopo la GRU.
            slots = self.gru(
                updates.reshape(-1, updates.shape[-1]),
                slots_prev.reshape(-1, slots_prev.shape[-1])
            )
            slots = slots.view(B, self.num_slots, -1)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class ObjectEncoder(nn.Module):
    """Proietta le caratteristiche della CNN ed estrae gli slot degli oggetti."""

    def __init__(self, num_slots=10, slot_dim=256, resolution=(7, 7)):
        """Configura proiezione convoluzionale, posizione e Slot Attention."""
        super().__init__()
        # Riduce i 512 canali ResNet alla dimensione latente degli slot.
        self.conv_proj = nn.Conv2d(512, slot_dim, kernel_size=1)
        self.pos_embed = SoftPositionEmbed(slot_dim, resolution)
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            in_dim=slot_dim,
            slot_dim=slot_dim
        )

    def forward(self, feat):
        """Converte la feature map in un insieme di slot oggetto."""
        # Porta i canali in coda perché posizione e Slot Attention lavorano su token.
        x = self.conv_proj(feat)
        x = x.permute(0, 2, 3, 1)  # B, H, W, D
        x = self.pos_embed(x)
        x = x.flatten(1, 2)  # B, H*W, D
        slots = self.slot_attention(x)
        return slots


class ObjectPresenceHead(nn.Module):
    """Predice se ogni slot corrisponde a un oggetto valido."""

    def __init__(self, slot_dim):
        """Inizializza il classificatore binario condiviso tra gli slot."""
        super().__init__()
        # La stessa trasformazione viene applicata indipendentemente a ogni slot.
        self.classifier = nn.Linear(slot_dim, 1)

    def forward(self, slots):
        """Restituisce un logit di presenza per ciascuno slot."""
        return self.classifier(slots).squeeze(-1)


class AttributeHead(nn.Module):
    """Predice gli attributi dell'oggetto per ogni slot."""

    def __init__(self, slot_dim):
        """Inizializza una testa per ogni vocabolario di attributi CLEVR."""
        super().__init__()
        # Le dimensioni di uscita coincidono con i vocabolari definiti nel dataset.
        self.color = nn.Linear(slot_dim, 8)
        self.material = nn.Linear(slot_dim, 2)
        self.shape = nn.Linear(slot_dim, 3)
        self.size = nn.Linear(slot_dim, 2)

    def forward(self, slots):
        """Calcola i logit di colore, materiale, forma e dimensione."""
        return {
            "attr_color": self.color(slots),
            "attr_material": self.material(slots),
            "attr_shape": self.shape(slots),
            "attr_size": self.size(slots),
        }


class CoordinateHead(nn.Module):
    """Regressione dei centri immagine normalizzati per ogni slot attivo."""

    def __init__(self, slot_dim):
        """Inizializza la MLP che regredisce le coordinate normalizzate."""
        super().__init__()
        # Due livelli sono sufficienti a produrre la coppia (x, y) per slot.
        self.regressor = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, 2),
        )

    def forward(self, slots):
        """Predice coordinate nell'intervallo unitario con una sigmoid."""
        return torch.sigmoid(self.regressor(slots))


class RelationalHead(nn.Module):
    """Predice le relazioni dirette per tutte le coppie di slot."""

    def __init__(self, slot_dim, num_relations=4):
        """Inizializza la MLP applicata alle coppie ordinate di slot."""
        super().__init__()
        # Concatena soggetto e oggetto prima della classificazione multi-label.
        self.num_relations = num_relations
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_relations)
        )

    def forward(self, slots):
        """Costruisce tutte le coppie e ne predice i logit relazionali."""
        B, N, D = slots.shape
        # Replica gli slot sui due assi per formare tutte le coppie ordinate N x N.
        subject_slots = slots.unsqueeze(2).expand(B, N, N, D)
        object_slots = slots.unsqueeze(1).expand(B, N, N, D)
        pair_features = torch.cat([subject_slots, object_slots], dim=-1)
        relation_logits = self.mlp(pair_features)
        # Espone l'output nel formato standard batch-canale-soggetto-oggetto.
        relations = relation_logits.permute(0, 3, 1, 2).contiguous()

        # Azzera la diagonale perché un oggetto non è in relazione con sé stesso.
        eye = torch.eye(N, device=slots.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        relations = relations.masked_fill(eye, 0.0)

        return relations


class GraphReasoningLayer(nn.Module):
    """Livello di message-passing sugli slot degli oggetti (ragionamento in stile GNN)."""

    def __init__(self, slot_dim):
        """Configura attenzione globale e aggiornamento residuo."""
        super().__init__()
        # Le proiezioni costruiscono messaggi globali tra tutte le coppie di slot.
        self.query = nn.Linear(slot_dim, slot_dim)
        self.key = nn.Linear(slot_dim, slot_dim)
        self.value = nn.Linear(slot_dim, slot_dim)
        self.scale = slot_dim ** -0.5

        # L'MLP fonde rappresentazione corrente e messaggio aggregato.
        self.update = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )
        self.norm = nn.LayerNorm(slot_dim)

    def forward(self, slots):
        """Propaga messaggi tra tutti gli slot con attenzione densa."""
        # Calcola attenzione completa tra slot e aggrega i relativi value.
        queries = self.query(slots)
        keys = self.key(slots)
        values = self.value(slots)

        attn = torch.einsum("bid,bjd->bij", queries.float(), keys.float()) * self.scale
        attn = F.softmax(attn, dim=-1).to(values.dtype)

        messages = torch.einsum("bij,bjd->bid", attn, values)
        # Usa connessione residua e LayerNorm per preservare l'informazione originale.
        updated = self.update(torch.cat([slots, messages], dim=-1))
        return self.norm(slots + updated)


class GraphReasoning(nn.Module):
    """Livelli di message-passing impilati per il ragionamento relazionale."""

    def __init__(self, slot_dim, num_layers=2):
        """Impila il numero richiesto di livelli di ragionamento denso."""
        super().__init__()
        # ModuleList registra correttamente i parametri di tutti i livelli.
        self.layers = nn.ModuleList([GraphReasoningLayer(slot_dim) for _ in range(num_layers)])

    def forward(self, slots):
        """Raffina gli slot attraversando sequenzialmente tutti i livelli."""
        x = slots
        # Ogni livello aggiorna l'intero insieme di slot.
        for layer in self.layers:
            x = layer(x)
        return x


class PyGGraphReasoning(nn.Module):
    """Backend di ragionamento su grafi di PyTorch Geometric che utilizza livelli GAT."""

    def __init__(self, slot_dim, num_layers=2):
        """Inizializza i livelli GAT e le rispettive normalizzazioni."""
        super().__init__()
        # Fallisce subito con un messaggio esplicito se il backend non è installato.
        if GATConv is None:
            raise ImportError(
                "torch_geometric is required for reasoning_backend='pyg'. "
                "Install PyG or use reasoning_backend='dense'."
            )
        # Mantiene invariata la dimensione usando una singola testa non concatenata.
        self.layers = nn.ModuleList([GATConv(slot_dim, slot_dim, heads=1, concat=False) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(slot_dim) for _ in range(num_layers)])

    @staticmethod
    def build_complete_edge_index(num_nodes, device):
        """Crea un grafo diretto completo escludendo gli auto-archi."""
        # Il prodotto cartesiano degli indici produce tutte le coppie possibili.
        node_ids = torch.arange(num_nodes, device=device)
        src = node_ids.repeat_interleave(num_nodes)
        dst = node_ids.repeat(num_nodes)
        valid = src != dst
        return torch.stack([src[valid], dst[valid]], dim=0)

    def forward(self, slots):
        """Applica i livelli GAT separatamente a ogni grafo del batch."""
        batch_size, num_slots, slot_dim = slots.shape
        # La topologia completa è identica per tutti gli elementi del batch.
        edge_index = self.build_complete_edge_index(num_slots, slots.device)
        outputs = []

        # GATConv riceve un singolo grafo; i grafi del batch sono quindi iterati.
        for batch_idx in range(batch_size):
            x = slots[batch_idx]
            for layer, norm in zip(self.layers, self.norms):
                # Combina aggiornamento GAT, ReLU, residuo e normalizzazione.
                updated = layer(x, edge_index)
                x = norm(x + F.relu(updated))
            outputs.append(x)

        return torch.stack(outputs, dim=0)


class VisualRelationModel(nn.Module):
    """Modello end-to-end per la predizione da immagine a grafo di scena."""

    def __init__(self, num_slots=10, reasoning_layers=2, reasoning_backend="dense", backbone_init="auto"):
        """Assembla backbone, encoder, reasoning e teste predittive."""
        super().__init__()
        # La prima metà estrae oggetti latenti e ne raffina le interazioni.
        self.backbone = CNNBackbone(backbone_init=backbone_init)
        self.object_encoder = ObjectEncoder(num_slots=num_slots)
        # Seleziona il backend mantenendo invariata l'interfaccia del modello.
        if reasoning_backend == "pyg":
            self.graph_reasoning = PyGGraphReasoning(slot_dim=256, num_layers=reasoning_layers)
        else:
            self.graph_reasoning = GraphReasoning(slot_dim=256, num_layers=reasoning_layers)
        # Le teste condividono gli stessi slot raffinati ma hanno obiettivi distinti.
        self.object_head = ObjectPresenceHead(slot_dim=256)
        self.attr_head = AttributeHead(slot_dim=256)
        self.coord_head = CoordinateHead(slot_dim=256)
        self.rel_head = RelationalHead(slot_dim=256, num_relations=4)

    def forward(self, images):
        """Esegue la pipeline e restituisce tutte le predizioni del grafo."""
        # Esegue in sequenza estrazione visiva, decomposizione e reasoning.
        feat = self.backbone(images)
        slots = self.object_encoder(feat)
        refined_slots = self.graph_reasoning(slots)

        # Calcola tutte le predizioni supervisionate a partire dagli slot comuni.
        object_logits = self.object_head(refined_slots)
        attr_logits = self.attr_head(refined_slots)
        pixel_coords = self.coord_head(refined_slots)
        relation_logits = self.rel_head(refined_slots)

        # Riunisce gli output in un dizionario stabile usato da loss e decoder.
        outputs = {
            "slots": refined_slots,
            "object_logits": object_logits,
            "pixel_coords": pixel_coords,
            "relations": relation_logits,
        }
        outputs.update(attr_logits)
        return outputs
