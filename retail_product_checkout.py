from flask import Flask, request, jsonify, send_file
import cv2 # type: ignore
import numpy as np
import pandas as pd
import torch # type: ignore
import os
from flask import send_from_directory
from collections import Counter
from reportlab.lib.pagesizes import letter # type: ignore
from reportlab.pdfgen import canvas # type: ignore
from io import BytesIO
from ultralytics import YOLO # type: ignore

from optimum.intel.openvino import OVModelOpenCLIPForZeroShotImageClassification, OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelOpenCLIPText, OVModelOpenCLIPVisual
#from nncf.quantization import GroupSizeFallbackMode
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from pathlib import Path
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 trained model
# model = YOLO('models/best.pt')

# Load billing data
billing_data = pd.read_csv('product_prices.csv')
billing_data.columns = ['Product', 'Price']

# Ensure upload directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# load pre-trained model
# ViT-H-14-378-quickgelu
#timm/vit_huge_patch14_clip_quickgelu_378.dfn5b
model = CLIPModel.from_pretrained("apple/DFN5B-CLIP-ViT-H-14-378")  #("openai/clip-vit-base-patch16")
# load preprocessor for model input
processor = CLIPProcessor.from_pretrained("apple/DFN5B-CLIP-ViT-H-14-378")  #("openai/clip-vit-base-patch16")

tokenizer = CLIPTokenizer.from_pretrained("apple/DFN5B-CLIP-ViT-H-14-378")


labels = [
    'Fruit/Apple/Golden-Delicious',
    'Fruit/Apple/Granny-Smith',
    'Fruit/Apple/Pink-Lady',
    'Fruit/Apple/Red-Delicious',
    'Fruit/Apple/Royal-Gala',
    'Fruit/Avocado',
    'Fruit/Banana',
    'Fruit/Kiwi',
    'Fruit/Lemon',
    'Fruit/Lime',
    'Fruit/Mango',
    'Fruit/Melon/Cantaloupe',
    'Fruit/Melon/Galia-Melon',
    'Fruit/Melon/Honeydew-Melon',
    'Fruit/Melon/Watermelon',
    'Fruit/Nectarine',
    'Fruit/Orange',
    'Fruit/Papaya',
    'Fruit/Passion-Fruit',
    'Fruit/Peach',
    'Fruit/Pear/Anjou',
    'Fruit/Pear/Conference',
    'Fruit/Pear/Kaiser',
    'Fruit/Pineapple',
    'Fruit/Plum',
    'Fruit/Pomegranate',
    'Fruit/Red-Grapefruit',
    'Fruit/Satsumas',
    'Packages/Juice/Bravo-Apple-Juice',
    'Packages/Juice/Bravo-Orange-Juice',
    'Packages/Juice/God-Morgon-Apple-Juice',
    'Packages/Juice/God-Morgon-Orange-Juice',
    'Packages/Juice/God-Morgon-Orange-Red-Grapefruit-Juice',
    'Packages/Juice/God-Morgon-Red-Grapefruit-Juice',
    'Packages/Juice/Tropicana-Apple-Juice',
    'Packages/Juice/Tropicana-Golden-Grapefruit',
    'Packages/Juice/Tropicana-Juice-Smooth',
    'Packages/Juice/Tropicana-Mandarin-Morning',
    'Packages/Milk/Arla-Ecological-Medium-Fat-Milk',
    'Packages/Milk/Arla-Lactose-Medium-Fat-Milk',
    'Packages/Milk/Arla-Medium-Fat-Milk',
    'Packages/Milk/Arla-Standard-Milk',
    'Packages/Milk/Garant-Ecological-Medium-Fat-Milk',
    'Packages/Milk/Garant-Ecological-Standard-Milk',
    'Packages/Oat-Milk/Oatly-Oat-Milk',
    'Packages/Oatghurt/Oatly-Natural-Oatghurt',
    'Packages/Sour-Cream/Arla-Ecological-Sour-Cream',
    'Packages/Sour-Cream/Arla-Sour-Cream',
    'Packages/Sour-Milk/Arla-Sour-Milk',
    'Packages/Soy-Milk/Alpro-Fresh-Soy-Milk',
    'Packages/Soy-Milk/Alpro-Shelf-Soy-Milk',
    'Packages/Soyghurt/Alpro-Blueberry-Soyghurt',
    'Packages/Soyghurt/Alpro-Vanilla-Soyghurt',
    'Packages/Yoghurt/Arla-Mild-Vanilla-Yoghurt',
    'Packages/Yoghurt/Arla-Natural-Mild-Low-Fat-Yoghurt',
    'Packages/Yoghurt/Arla-Natural-Yoghurt',
    'Packages/Yoghurt/Valio-Vanilla-Yoghurt',
    'Packages/Yoghurt/Yoggi-Strawberry-Yoghurt',
    'Packages/Yoghurt/Yoggi-Vanilla-Yoghurt',
    'Packages/Instant-Noodles/Nissin-Premium-Ramen',
    'Packages/Chips/Snek-Mi-Mi',
    'Vegetables/Asparagus',
    'Vegetables/Aubergine',
    'Vegetables/Brown-Cap-Mushroom',
    'Vegetables/Cabbage',
    'Vegetables/Carrots',
    'Vegetables/Cucumber',
    'Vegetables/Garlic',
    'Vegetables/Ginger',
    'Vegetables/Leek',
    'Vegetables/Onion/Yellow-Onion',
    'Vegetables/Pepper/Green-Bell-Pepper',
    'Vegetables/Pepper/Orange-Bell-Pepper',
    'Vegetables/Pepper/Red-Bell-Pepper',
    'Vegetables/Pepper/Yellow-Bell-Pepper',
    'Vegetables/Potato/Floury-Potato',
    'Vegetables/Potato/Solid-Potato',
    'Vegetables/Potato/Sweet-Potato',
    'Vegetables/Red-Beet',
    'Vegetables/Tomato/Beef-Tomato',
    'Vegetables/Tomato/Regular-Tomato',
    'Vegetables/Tomato/Vine-Tomato',
    'Vegetables/Zucchini',
    'Beverages/Can/Milo',
    'Beverages/Can/Coke'
]

device = 'GPU'
model_id = "apple/DFN5B-CLIP-ViT-H-14-378" #"openai/clip-vit-base-patch16"

to_quantize = False
model_base_dir = Path(f"{model_id.split('/')[-1]}-openclip")
additional_args = {}
zeroshot_weights = None

if to_quantize:
    model_dir = model_base_dir / "INT8"
    if not model_dir.exists():
        OVModelOpenCLIPForZeroShotImageClassification.from_pretrained("apple/DFN5B-CLIP-ViT-H-14-378", #"openai/clip-vit-base-patch16"
            quantization_config=OVWeightQuantizationConfig(bits=8)
        ).save_pretrained(model_dir)
else:
    model_dir = model_base_dir / "FP16"
    if not model_dir.exists():
        OVModelOpenCLIPForZeroShotImageClassification.from_pretrained("apple/DFN5B-CLIP-ViT-H-14-378").save_pretrained(model_dir)

#labels = ['cat', 'tiger']
text_descriptions = [
     [f"a photo of a {label}.",
      f"a product photo of a {label}.",
      f"a retail image of a {label}.",
      f"a picture of a {label}.",
      f"an image of a {label}.",
      f"{label} for sale.",
      f"a shelf photo of a {label}.",
      f"a close-up photo of a {label}."] for label in labels]

class_templates =  [
      "a photo of a {label}.",
      "a product photo of a {label}.",
      "a retail image of a {label}.",
      "a picture of a {label}.",
      "an image of a {label}.",
      "{label} for sale.",
      "a shelf photo of a {label}.",
      "a close-up photo of a {label}."]

zeroshot_weights_pth = Path("clip_zeroshot_cls.pth")

if zeroshot_weights_pth.exists():
   zeroshot_weights = torch.load("clip_zeroshot_cls.pth", map_location="cpu")

   print(f"zeroshot_weights's shape: {zeroshot_weights.shape}")
   print(f"zeroshot_weights: {zeroshot_weights}")
   
else:
    zeroshot_weights = []
    for label in tqdm(labels):
        texts = [template.format(label=label) for template in class_templates]
        t1 = time.time()
        #texts = tokenizer(text=texts, return_tensors="pt", padding=True)
        texts = processor(text=texts, return_tensors="pt", padding=True)
        #print(f"input_ids: {texts}")
        with torch.no_grad():
            class_embeddings = model.get_text_features(**texts)

        t2=time.time()
        print(f"elapsed time (text encode): {t2-t1} s")
        
        #class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

        print(f"class_embeddings shape: {class_embedding.shape}")
        print(f"class_embeddings: {class_embedding}")

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    print(f"zeroshot_weights's shape: {zeroshot_weights.shape}")
    print(f"zeroshot_weights: {zeroshot_weights}")
    
    torch.save(zeroshot_weights, "clip_zeroshot_cls.pth")

@app.route('/')
def home():
    return send_file(os.path.join(BASE_DIR, 'templates/webpage_design.html'))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, file.filename) # type: ignore
        file.save(image_path)

        img = cv2.imread(image_path)

        # Run YOLOv8 detection
        #results = model(img)
        #result = results[0]
        
        t1 = time.time()
        img_inputs = processor(images=[img], return_tensors="pt", padding=True)

        t3 = time.time()
        ov_model_vision = OVModelOpenCLIPVisual.from_pretrained(model_dir, device=device)
        visual_features = ov_model_vision(**img_inputs)

        t2 = time.time()
        print(f"visual_features: {visual_features}, shape: {visual_features['image_features'].shape}")

        logits = 100. * visual_features['image_features'] @ zeroshot_weights
        logits_top5 = logits.topk(5, 1, True, True)[1].squeeze().tolist()
        logits_top5_values = logits.topk(5, 1, True, True)[0].squeeze().tolist()

        top5_label = [labels[x] for x in logits_top5]
        pred = logits.argmax(axis=1)

        print(f"elapsed time (pipeline): {t2-t1}s")
        print(f"elapsed time (image encode): {t2-t3}s")
        print(f"pred: {pred}")
        print(f"top-5 pred: {top5_label}")

        # Get class indices and map to names
        #class_ids = result.boxes.cls.cpu().numpy().astype(int)
        #detected_products = [model.names[i] for i in class_ids]
        detected_products = [labels[n].split('/')[-1] for n in pred.tolist()]
        print(f"detected_products: {detected_products}")
        # Generate bill
        bill = []
        total_price = 0
        print(f"detected_products: {detected_products}")
        print(f"product_counts: {detected_products}")
        
        # Count occurrences of each product
        product_counts = Counter(detected_products)
        print(f"product_counts: {product_counts}")
        
        for product, count in product_counts.items():
            product_info = billing_data[billing_data['Product'] == product]
            if not product_info.empty:
                unit_price = float(product_info['Price'].values[0])
                total = unit_price * count
                bill.append({
                    'Product': product,
                    'Quantity': count,
                    'Unit_Price': unit_price,
                    'Total': total
                })
                total_price += total

        response = {'detected_products': detected_products, 'bill': bill, 'total_price': total_price, 'total_items': sum(product_counts.values())}
        return jsonify(response)

    except Exception as e:
        print("Error during upload_image:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/payment')
def payment_page():
    amount = request.args.get('amount', '0.00')
    return send_file(os.path.join(BASE_DIR, 'templates/payment.html'))


@app.route('/payment/success', methods=['GET'])
def payment_success():
    return send_file(os.path.join(BASE_DIR, 'templates/success.html'))

@app.route('/download_bill', methods=['POST'])
def download_bill():
    data = request.get_json()

    bill = data.get('bill', [])
    total_price = data.get('total_price', 0)
    total_items = data.get('total_items', 0)

    # Generate PDF
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Smart Retail Checkout - Final Bill")
    y -= 40

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Product")
    c.drawString(200, y, "Qty")
    c.drawString(250, y, "Unit Price")
    c.drawString(350, y, "Total")
    y -= 20

    c.setFont("Helvetica", 12)
    for item in bill:
        c.drawString(50, y, item['Product'])
        c.drawString(200, y, str(item['Quantity']))
        c.drawString(250, y, f"${item['Unit_Price']:.2f}")
        c.drawString(350, y, f"${item['Total']:.2f}")
        y -= 20

    y -= 10
    c.line(50, y, 500, y)
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Total Items: {total_items}")
    c.drawString(250, y, f"Total Price: ${total_price:.2f}")

    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="final_bill.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    print("Server is starting...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
