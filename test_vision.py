from pathlib import Path
from vision import describe_image, VISION_MODEL


print("Vision test başladı...")
print("Kullanılan model:", VISION_MODEL)

BASE_DIR = Path(__file__).parent
image_path = BASE_DIR / "screenshots" / "test.jpg"

print("Görsel yolu:", image_path)
print("Görsel var mı:", image_path.exists())

if not image_path.exists():
    raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

print("Model çağrılıyor...")

result = describe_image(
    image_path=str(image_path),
    question="Bu görselde ne olduğunu Türkçe açıkla. Görselde yazı varsa aynen oku."
)

print("Model cevabı geldi.")
print("Cevap repr hali:")
print(repr(result))

print("Normal cevap:")
print(result)