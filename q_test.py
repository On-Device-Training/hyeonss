import torch
import torch.nn as nn
import os
import random
import copy
import argparse
import torch.nn.functional as F
import torchao
from torch.quantization import prepare, convert, prepare_qat
from torch.ao.quantization import QConfig
from torch.ao.quantization import quantize
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

from face_model import EdgeNeXt
# from casia import CASIAWebFaceDataset, create_dummy_dataset

# Backend
torch.backends.quantized.engine = 'fbgemm'
print("Current quantized engine:", torch.backends.quantized.engine)

# EdgeNeXt Quantization wrapper
class QuantizableEdgeNeXt(EdgeNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()
        self._prepare_for_quantization()
    
    def _prepare_for_quantization(self):
        self._prepare_attention_modules(self)

    def _prepare_attention_modules(self, module):
        for name, child in module.named_children():
            child.qconfig = get_edgenext_qconfig()
            self._prepare_attention_modules(child)
    
    def forward(self, x):
        # x = self.quant(x)
        x = super().forward(x)
        # x = self.dequant(x)
        return x

# Original EdgeNeXt
def get_original_edgenext_model(num_classes=10575, pretrained_path=None):
    model = QuantizableEdgeNeXt(
        in_chans=3,
        num_classes=num_classes,
        dims=(24, 48, 88, 168),
        depths=(2, 2, 6, 2),
        global_block_counts=(0, 1, 1, 1),
        kernel_sizes=(3, 5, 7, 9),
        heads=(4, 4, 4, 4),
        d2_scales=(2, 2, 3, 4),
        use_pos_emb=(False, True, False, False),
        ls_init_value=1e-6,
        expand_ratio=4,
        conv_bias=True,
        act_layer=nn.GELU,
        drop_path_rate=0.,
        drop_rate=0.,
    )
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    return model

# Custom QConfig
def get_edgenext_qconfig():
    return QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=True
        ),
        weight=MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    )

# Dynamic Quantization
def quantize_edgenext_dynamic_advanced(model):
    try:
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        layers_to_quantize = {nn.Linear, nn.Conv2d}
        quantized_model = torch.quantization.quantize_dynamic(model_copy, layers_to_quantize, dtype=torch.qint8)
        print("Dynamic quantization complete")
        return quantized_model
    except Exception as e:
        print(f"Dynamic quantization failed: {e}")
        return None

# PTQ
def quantize_edgenext_ptq_advanced(model, calib_loader, num_calibration_batches=500):
    try:
        model_copy = copy.deepcopy(model)
        model_copy.to('cpu').eval()
        model_copy.qconfig = get_edgenext_qconfig()

        # def set_module_qconfig(module, qconfig):
        #     for _, child in module.named_children():
        #         child.qconfig = qconfig
        #         set_module_qconfig(child, qconfig)

        # def set_module_qconfig(module, qconfig):
        #     if isinstance(module, (nn.Linear, nn.Conv2d)):
        #         module.qconfig = qconfig
        #     else:
        #         module.qconfig = None
        #     for child in module.children():
        #         set_module_qconfig(child, qconfig)
        def set_module_qconfig(module, qconfig):
            module.qconfig = qconfig
            for child in module.children():
                set_module_qconfig(child, qconfig)

            
        set_module_qconfig(model_copy, get_edgenext_qconfig())

        model_prepared = prepare(model_copy, inplace=False)

        # Calibration
        calibration_count = 0
        with torch.no_grad():
            for i, batch_data in enumerate(calib_loader):
                if calibration_count >= num_calibration_batches:
                    break
                try:
                    if isinstance(batch_data, (list, tuple)):
                        images, _ = batch_data
                    else:
                        images = batch_data
                    if images.dim() == 3:
                        images = images.unsqueeze(0)
                    if images.shape[-2:] != (224, 224):
                        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

                    images = images.to('cpu').float()
                    _ = model_prepared(images)
                    calibration_count += 1
                except:
                    continue

        model_quantized = convert(model_prepared, inplace=False)
        print("PTQ quantization complete")
        return model_quantized
    except Exception as e:
        print(f"PTQ failed: {e}")
        return None
    
def quantize_edgenext_qat_advanced(model, train_loader, num_epochs=1, lr=1e-4):
    """
    QAT (Quantization Aware Training) for EdgeNeXt
    """
    try:
        print("Starting Quantization Aware Training (QAT)...")
        model_copy = copy.deepcopy(model).to("cpu")
        model_copy.train()
        model_copy.qconfig = get_edgenext_qconfig()

        def set_module_qconfig(module, qconfig):
            for _, child in module.named_children():
                child.qconfig = qconfig
                set_module_qconfig(child, qconfig)
        set_module_qconfig(model_copy, get_edgenext_qconfig())

        model_prepared = prepare_qat(model_copy, inplace=False)
        print("Model prepared for QAT.")

        optimizer = torch.optim.Adam(model_prepared.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model_prepared.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

            for i, (images, labels) in enumerate(progress_bar):
                optimizer.zero_grad()
                outputs = model_prepared(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                avg_loss = running_loss / (i + 1)
                progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})
            
            print(f"Epoch {epoch+1}/{num_epochs} complete. Avg Loss: {running_loss / len(train_loader):.4f}")
            

        model_quantized = convert(model_prepared.eval(), inplace=False)
        print("QAT quantization complete")
        return model_quantized

    except Exception as e:
        print(f"QAT failed: {e}")
        return None
  
def quantize_edgenext_fp16(model):
    try:
        print("FP16 quantization in progress...")
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.half() 
        print("FP16 quantization complete.")
        return model_copy
    except Exception as e:
        print(f"FP16 quantization failed: {e}")
        return None

def quantize_edgenext_torchao(model):
    try:
        print("TorchAO quantization in progress...")
        try:
            model_copy = copy.deepcopy(model)
            model_copy.eval()
            
            # TorchAO dynamic quantization
            config = Int8DynamicActivationInt8WeightConfig()
            quantize_(model_copy, config)
            print("TorchAO quantization complete.")
            return model_copy
            
        except ImportError:
            print("TorchAO not available, falling back to dynamic quantization")
            return quantize_edgenext_dynamic_advanced(model)
            
    except Exception as e:
        print(f"TorchAO quantization failed: {e}")
        return None
    
def get_model_size(model):
    temp_file = "temp_model_file.pth"
    try:
        torch.save(model.state_dict(), temp_file)
        size_bytes = os.path.getsize(temp_file)
        os.remove(temp_file)
        return size_bytes / (1024 * 1024)
    except:
        return 0

def get_num_parameters(model):
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    except:
        return 0, 0

def evaluate_model_safe(model, data_loader, max_batches=None, show_progress=True):
    if model is None:
        return 0
    model.eval()
    correct, total = 0, 0
    iterator = data_loader
    if show_progress:
        desc = f"Evaluating {model.__class__.__name__}"
        iterator = tqdm(data_loader, desc=desc)

    with torch.no_grad():
        for i, batch_data in enumerate(iterator):
            if max_batches is not None and i >= max_batches:
                break
            try:
                if isinstance(batch_data, (list, tuple)):
                    images, labels = batch_data
                else:
                    images = batch_data
                    labels = torch.zeros(images.size(0), dtype=torch.long)
                if images.shape[-1] != 224 or images.shape[-2] != 224:
                    images = transforms.Resize((224, 224))(images)
                outputs = model(images)
                if hasattr(outputs, "dequantize"):
                    outputs = outputs.dequantize()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except:
                continue
    return 100 * correct / total if total > 0 else 0

def log_result(title, model, data_loader, file_path="model_stats.txt", show_progress=True):
    total_params, trainable_params = get_num_parameters(model)
    size = get_model_size(model)
    acc = evaluate_model_safe(model, data_loader, show_progress=show_progress)
    result = f"""
{title}:
  - Total Parameters: {total_params:,}
  - Trainable Parameters: {trainable_params:,}
  - Model Size: {size:.2f} MB
  - Test Accuracy: {acc:.2f}%
{'-'*60}
"""
    with open(file_path, "a", encoding='utf-8') as f:
        f.write(result)
    print(f"{title} - Total: {total_params:,}, Size: {size:.2f}MB, Acc: {acc:.2f}%")

def save_model_structure(model, file_path="model_structure.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Model structure: {model.__class__.__name__}\n")
        f.write("="*60 + "\n")
        f.write(str(model))
        f.write("\n\nDetailed modules:\n")
        f.write("-"*60 + "\n")
        for name, module in model.named_modules():
            f.write(f"{name}: {module}\n")
    print(f"Model structure saved to {file_path}")

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, nargs='+', default=["original"],
                        choices=["original", "dynamic", "ptq", "fp16", "torchao", "qat"],
                        help="Quantization type to use")
    args = parser.parse_args()
    PRETRAINED_PATH = "edgeface/checkpoints/pruned_yai_0_5.pt"
    PRUNED_MODEL_PATH = None
    data_dir = "data_set/casia-webface1"
    list_file = os.path.join(data_dir, "subset_list.txt")

    with open("model_stats.txt", "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EdgeNeXt Quantization Results\n")
        f.write("="*60 + "\n")

    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    try:
        if os.path.exists(list_file):
            full_dataset = CASIAWebFaceDataset(root_dir=data_dir, list_file=list_file, transform=transform)

            NUM_CLASSES = max(label for _, label in full_dataset.samples) + 1

            total_len = len(full_dataset)
            val_len = int(total_len * 0.2)
            train_len = total_len - val_len

            train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

            num_calib_samples = min(200, len(val_dataset))
            calib_indices = random.sample(range(len(val_dataset)), num_calib_samples)
            calib_subset = Subset(val_dataset, calib_indices)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            calib_loader = DataLoader(calib_subset, batch_size=4, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
            

            print(f"Real dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Calib: {len(calib_subset)}, NUM_CLASSES={NUM_CLASSES}")
        else:
            raise FileNotFoundError(f"Dataset not found at '{data_dir}'")
    except:
        NUM_CLASSES = 512
        calib_loader = create_dummy_dataset(100, NUM_CLASSES)
        val_loader = create_dummy_dataset(50, NUM_CLASSES)
        print("Using dummy dataset for evaluation.")

    # Original Model
    original_model = get_original_edgenext_model(NUM_CLASSES, PRETRAINED_PATH)
    
    # for quant_type in args.modes:
    #     quant_type_lower = quant_type.lower()
    #     print(f"\n=== Evaluating {quant_type_lower.upper()} model ===")
        
    #     if quant_type_lower == "original":
    #         model_to_eval = original_model
    #         title = "Original EdgeNeXt Model"
    #     elif quant_type_lower == "dynamic":
    #         model_to_eval = quantize_edgenext_dynamic_advanced(copy.deepcopy(original_model))
    #         title = "Dynamic Quantized EdgeNeXt"
    #     elif quant_type_lower == "ptq":
    #         model_to_eval = quantize_edgenext_ptq_advanced(copy.deepcopy(original_model), calib_loader, num_calibration_batches=500)
    #         title = "PTQ Quantized EdgeNeXt"
    #         save_model_structure(model_to_eval, "ptq_model_structure.txt")
    #     elif quant_type_lower == "qat":
    #         model_to_eval = quantize_edgenext_qat_advanced(copy.deepcopy(original_model), train_loader=train_loader, num_epochs=3)
    #         title = "QAT Quantized EdgeNeXt"
    #     elif quant_type_lower == "fp16":
    #         model_to_eval = quantize_edgenext_fp16(copy.deepcopy(original_model))
    #         title = "FP16 EdgeNeXt"
    #     elif quant_type_lower == "torchao":
    #         model_to_eval = quantize_edgenext_torchao(copy.deepcopy(original_model))
    #         title = "TorchAO Quantized EdgeNeXt"
    #     else:
    #         print(f"Unknown quant_type: {quant_type_lower}, skipping.")
    #         continue

    for quant_type in args.modes:
        quant_type_lower = quant_type.lower()
        print(f"\n=== Processing {quant_type_lower.upper()} model ===")
        
        if quant_type_lower == "original":
            model_to_save = original_model
            file_name = "edgenext_original.pt"
        elif quant_type_lower == "dynamic":
            model_to_save = quantize_edgenext_dynamic_advanced(copy.deepcopy(original_model))
            file_name = "edgenext_dynamic.pt"
        elif quant_type_lower == "ptq":
            model_to_save = quantize_edgenext_ptq_advanced(copy.deepcopy(original_model), calib_loader, num_calibration_batches=500)
            file_name = "edgenext_ptq.pt"
            save_model_structure(model_to_save, "ptq_model_structure.txt")
        elif quant_type_lower == "qat":
            model_to_save = quantize_edgenext_qat_advanced(copy.deepcopy(original_model), train_loader=train_loader, num_epochs=3)
            file_name = "edgenext_qat.pt"
        elif quant_type_lower == "fp16":
            model_to_save = quantize_edgenext_fp16(copy.deepcopy(original_model))
            file_name = "edgenext_fp16.pt"
        elif quant_type_lower == "torchao":
            model_to_save = quantize_edgenext_torchao(copy.deepcopy(original_model))
            file_name = "edgenext_torchao.pt"
        else:
            print(f"Unknown quant_type: {quant_type_lower}, skipping.")
            continue

        torch.save(model_to_save.state_dict(), file_name)
        print(f"{quant_type_lower.upper()} model saved to {file_name}")


        # print(f"Running evaluation for {title} ...")
        # log_result(title, model_to_eval, val_loader, show_progress=True)
        # print(f"{title} evaluation done. Results saved in 'model_stats.txt'")

    print("All tests completed. Results saved in 'model_stats.txt'")
