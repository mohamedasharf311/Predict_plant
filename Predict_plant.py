import torch
from torchvision import transforms, models
from PIL import Image

# 🔹 المسارات
save_path = "/content/drive/MyDrive/plant_model.pth"  # الموديل المدرب
data_dir = "/content/drive/MyDrive/planet/Plant_Data/Plant_Data/train"  # فولدر الفئات

# 🔹 إعداد الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 التحويلات للصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔹 تحميل الموديل
model = models.resnet18(pretrained=True)
# عدد الفئات بناءً على الفولدرات الموجودة
classes = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

checkpoint = torch.load(save_path, map_location=device)
model.load_state_dict(checkpoint['model_state'], strict=False)
model.eval()

print(f"✅ الموديل جاهز للتنبؤ على {len(classes)} فئات.")

# 🔹 دالة للتنبؤ بصورة واحدة
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ الصورة تالفة أو غير صالحة: {image_path}")
        return

    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
        print(f"📸 الصورة: {image_path}")
        print(f"✅ الفئة المتوقعة: {class_name}")

# 🔹 طلب صورة من المستخدم
image_path = input("🖼️ ضع مسار الصورة للتنبؤ: ")
predict_image(image_path)
