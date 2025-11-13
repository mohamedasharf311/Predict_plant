import os
import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from google.colab import drive

# 🔗 ربط Google Drive
drive.mount('/content/drive', force_remount=True)

# 📁 المسارات
data_dir = "/content/drive/MyDrive/planet/Plant_Data/Plant_Data/train"
save_path = "/content/drive/MyDrive/plant_model.pth"
log_file = "/content/drive/MyDrive/trained_folders.txt"

# 🌱 التحويلات للصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔹 قراءة الفولدرات التي تم تدريبها مسبقًا
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        trained_folders = f.read().splitlines()
else:
    trained_folders = []

# 🔹 جمع كل الفولدرات الجديدة
all_folders = sorted(os.listdir(data_dir))
new_folders = [f for f in all_folders if f not in trained_folders]
print(f"📂 عدد الفولدرات الجديدة: {len(new_folders)}")

# 🔹 اختيار 10 فولدرات جديدة فقط
batch_size = 10
batch_folders = new_folders[:batch_size]

if not batch_folders:
    print("✅ لا توجد فولدرات جديدة للتدريب!")
else:
    print("🎯 سيتم التدريب على:", batch_folders)

    # 🔹 التأكد من وجود صور صالحة داخل الفولدرات
    valid_folders = []
    for folder in batch_folders:
        folder_path = os.path.join(data_dir, folder)
        valid_images = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    valid_images.append(os.path.join(root, f))
        if valid_images:
            valid_folders.append(folder)
        else:
            print(f"⚠️ تخطي الفولدر الفارغ: {folder}")

    if not valid_folders:
        print("⚠️ كل الفولدرات في هذه الدفعة فارغة أو غير صالحة!")
    else:

        # 🔹 Safe Loader لتجاهل الصور التالفة
        def pil_loader_safe(path):
            try:
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            except Exception:
                print(f"⚠️ تجاهل الصورة التالفة: {path}")
                # إرجاع صورة سوداء صغيرة بدل الصورة التالفة
                return Image.new('RGB', (224, 224))

        class SafeImageFolder(ImageFolder):
            def __getitem__(self, index):
                path, target = self.samples[index]
                sample = pil_loader_safe(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target

        # 🔹 تحميل البيانات
        full_dataset = SafeImageFolder(data_dir, transform=transform)
        # 🔹 فلترة Dataset ليتضمن فقط valid_folders
        valid_idx = [full_dataset.class_to_idx[f] for f in valid_folders]
        full_dataset.samples = [s for s in full_dataset.samples if s[1] in valid_idx]
        full_dataset.targets = [s[1] for s in full_dataset.samples]
        train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=True)

        # 🔹 إنشاء أو تحميل الموديل
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
        model = model.to(device)

        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'], strict=False)
            print("♻️ تم تحميل الموديل القديم ومتابعة التدريب.")

        # 🔹 التدريب
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 2

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # 🔹 حفظ الموديل
        torch.save({'model_state': model.state_dict()}, save_path)
        print("💾 تم حفظ النموذج بعد التدريب على هذه الدفعة!")

        # 🔹 تحديث سجل الفولدرات المتدربة
        with open(log_file, "a") as f:
            for folder in valid_folders:
                f.write(folder + "\n")

        print("✅ تم تدريب الفولدرات:", valid_folders)
