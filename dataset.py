"""Carica CLEVR e converte scene, immagini e relazioni in tensori supervisionati."""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ClevrRelationDataset(Dataset):
    """Wrapper del dataset CLEVR per la predizione dei grafi di scena.

    Ogni campione include:
    - Tensore dell'immagine.
    - Maschera dell'oggetto a livello di slot.
    - Attributi a livello di slot (colore, materiale, forma, dimensione).
    - Matrici delle relazioni dirette a coppie.
    """

    # I vocabolari fissano la corrispondenza stabile tra etichette testuali e indici.
    COLOR_VOCAB = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    MATERIAL_VOCAB = ["rubber", "metal"]
    SHAPE_VOCAB = ["cube", "sphere", "cylinder"]
    SIZE_VOCAB = ["small", "large"]
    RELATION_NAMES = ["left", "right", "front", "behind"]
    # Le coordinate CLEVR sono espresse rispetto alla risoluzione originale 480 x 320.
    ORIGINAL_IMAGE_WIDTH = 480.0
    ORIGINAL_IMAGE_HEIGHT = 320.0

    def __init__(self, img_dir, scene_json, max_objects=10, max_scenes=None):
        """Inizializza il dataset dai file delle immagini e delle scene di CLEVR."""
        # Conserva la directory delle immagini e il limite massimo di slot.
        self.img_dir = img_dir
        self.max_objects = max_objects

        # Precalcola i dizionari per codificare gli attributi in tempo costante.
        self.color_to_idx = {name: idx for idx, name in enumerate(self.COLOR_VOCAB)}
        self.material_to_idx = {name: idx for idx, name in enumerate(self.MATERIAL_VOCAB)}
        self.shape_to_idx = {name: idx for idx, name in enumerate(self.SHAPE_VOCAB)}
        self.size_to_idx = {name: idx for idx, name in enumerate(self.SIZE_VOCAB)}

        # Carica in memoria le descrizioni strutturate delle scene CLEVR.
        with open(scene_json, "r") as f:
            scenes = json.load(f)["scenes"]

        # Applica un eventuale limite utile per esperimenti rapidi e debug.
        if max_scenes is not None:
            self.scenes = scenes[:max_scenes]
        else:
            self.scenes = scenes

        # Uniforma risoluzione e normalizzazione ai valori attesi da ResNet34.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """Restituisce il numero di scene disponibili nel dataset."""
        return len(self.scenes)

    def _resolve_image_path(self, image_filename):
        """Risolve il percorso dell'immagine per i layout CLEVR con sottocartelle di split."""
        # Prima prova il layout piatto, accettato per dataset personalizzati.
        direct_path = os.path.join(self.img_dir, image_filename)
        if os.path.isfile(direct_path):
            return direct_path

        # Deduce lo split dal nome standard CLEVR quando sono presenti sottocartelle.
        filename_lower = image_filename.lower()
        split_folder = None
        if "train" in filename_lower:
            split_folder = "train"
        elif "val" in filename_lower:
            split_folder = "val"
        elif "test" in filename_lower:
            split_folder = "test"

        # Verifica per prima la sottocartella dedotta, evitando ricerche superflue.
        if split_folder is not None:
            split_path = os.path.join(self.img_dir, split_folder, image_filename)
            if os.path.isfile(split_path):
                return split_path

        # Come fallback cerca il file in tutti gli split noti.
        for candidate_split in ("train", "val", "test"):
            candidate_path = os.path.join(self.img_dir, candidate_split, image_filename)
            if os.path.isfile(candidate_path):
                return candidate_path

        # Espone tutti i percorsi tentati per rendere l'errore diagnosticabile.
        raise FileNotFoundError(
            f"Image file not found for '{image_filename}'. "
            f"Checked: {direct_path}, {os.path.join(self.img_dir, 'train', image_filename)}, "
            f"{os.path.join(self.img_dir, 'val', image_filename)}, {os.path.join(self.img_dir, 'test', image_filename)}"
        )

    def __getitem__(self, idx):
        """Carica una scena e costruisce immagine trasformata e target supervisionati."""
        # Recupera metadati e immagine corrispondenti all'indice richiesto.
        scene = self.scenes[idx]
        img_path = self._resolve_image_path(scene["image_filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Alloca target a dimensione fissa; gli slot vuoti restano mascherati.
        N = self.max_objects
        object_mask = torch.zeros(N, dtype=torch.float32)

        # -100 è l'indice ignorato dalla cross-entropy per gli slot senza oggetto.
        attr_color = torch.full((N,), fill_value=-100, dtype=torch.long)
        attr_material = torch.full((N,), fill_value=-100, dtype=torch.long)
        attr_shape = torch.full((N,), fill_value=-100, dtype=torch.long)
        attr_size = torch.full((N,), fill_value=-100, dtype=torch.long)
        # Le due coordinate saranno normalizzate nell'intervallo [0, 1].
        pixel_coords = torch.zeros((N, 2), dtype=torch.float32)

        # Codifica soltanto i primi N oggetti, coerentemente con il numero di slot.
        for i, obj in enumerate(scene.get("objects", [])[:N]):
            object_mask[i] = 1.0
            attr_color[i] = self.color_to_idx[obj["color"]]
            attr_material[i] = self.material_to_idx[obj["material"]]
            attr_shape[i] = self.shape_to_idx[obj["shape"]]
            attr_size[i] = self.size_to_idx[obj["size"]]
            pixel_x, pixel_y = obj["pixel_coords"][:2]
            pixel_coords[i, 0] = float(pixel_x) / self.ORIGINAL_IMAGE_WIDTH
            pixel_coords[i, 1] = float(pixel_y) / self.ORIGINAL_IMAGE_HEIGHT

        # Ogni canale contiene la matrice di adiacenza diretta di una relazione.
        relation_labels = torch.zeros(len(self.RELATION_NAMES), N, N, dtype=torch.float32)

        # Trasforma le liste di indici CLEVR in matrici binarie soggetto-oggetto.
        for rel_idx, rel_name in enumerate(self.RELATION_NAMES):
            for i, rel_list in enumerate(scene["relationships"][rel_name]):
                for j in rel_list:
                    if i < N and j < N:
                        relation_labels[rel_idx, i, j] = 1

        # Raggruppa tutte le supervisioni in un dizionario compatibile col DataLoader.
        targets = {
            "object_mask": object_mask,
            "attr_color": attr_color,
            "attr_material": attr_material,
            "attr_shape": attr_shape,
            "attr_size": attr_size,
            "pixel_coords": pixel_coords,
            "relations": relation_labels,
            "image_filename": scene["image_filename"],
        }

        # Restituisce la coppia standard input-target usata da training e valutazione.
        return image, targets
