import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DogClassifierService:
    def __init__(self):
        self.model = None
        self.ensemble_models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None
        self.class_names = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info("Loading dog-optimized model...")
            
            models_to_try = [
                ('efficientnet_b4', 'EfficientNet-B4 (Best for fine-grained classification)'),
                ('efficientnet_b3', 'EfficientNet-B3 (Good for fine-grained classification)'),
                ('efficientnet_b2', 'EfficientNet-B2 (Balanced performance)'),
                ('resnet152', 'ResNet-152 (Reliable for classification)'),
                ('densenet201', 'DenseNet-201 (Good feature extraction)'),
                ('efficientnet_b1', 'EfficientNet-B1 (Fast and accurate)'),
                ('efficientnet_b0', 'EfficientNet-B0 (Lightweight)')
            ]
            
            for model_name, description in models_to_try:
                try:
                    self.model = timm.create_model(model_name, pretrained=True)
                    self.current_model_name = model_name
                    logger.info(f"{description} loaded successfully")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("No suitable model could be loaded")
            
            self.model.eval()
            self.model.to(self.device)

            if 'swin_large_patch4_window12_384' in self.current_model_name:

                self.transform = transforms.Compose([
                    transforms.Resize((384, 384)), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            elif 'swin_base_patch4_window7_224' in self.current_model_name:
           
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            elif 'convnext' in self.current_model_name:
        
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            elif 'efficientnet' in self.current_model_name:
               
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
           
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            self._load_dog_breed_classes()
            self._load_ensemble_models()
            
            logger.info(f"Dog-optimized model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_ensemble_models(self):
        try:
            logger.info("📝 Skipping ensemble models to avoid input size conflicts")
            return
            
            ensemble_candidates = [
                ('efficientnet_b0', 'EfficientNet-B0'),
                ('resnet50', 'ResNet-50'),
                ('densenet121', 'DenseNet-121')
            ]
            
            for model_name, description in ensemble_candidates:
                try:
                    if model_name != getattr(self, 'current_model_name', ''):
                        ensemble_model = timm.create_model(model_name, pretrained=True)
                        ensemble_model.eval()
                        ensemble_model.to(self.device)
                        self.ensemble_models.append((ensemble_model, description))
                        logger.info(f"✅ Ensemble model loaded: {description}")
                        
                        if len(self.ensemble_models) >= 2:
                            break
                except Exception as e:
                    logger.warning(f"Failed to load ensemble model {model_name}: {e}")
                    continue
                    
            if self.ensemble_models:
                logger.info(f"🎯 Loaded {len(self.ensemble_models)} ensemble models for improved accuracy")
            else:
                logger.info("📝 No ensemble models loaded - using single model")
                
        except Exception as e:
            logger.warning(f"Ensemble model loading failed: {e}")
    
    def _load_dog_breed_classes(self):
        self.imagenet_dog_classes = {
            151: 'Chihuahua',
            152: 'Japanese spaniel', 
            153: 'Maltese dog',
            154: 'Pekinese',
            155: 'Shih-Tzu',
            156: 'Blenheim spaniel',
            157: 'papillon',
            158: 'toy terrier',
            159: 'Rhodesian ridgeback',
            160: 'Afghan hound',
            161: 'basset',
            162: 'beagle',
            163: 'bloodhound',
            164: 'bluetick',
            165: 'black-and-tan coonhound',
            166: 'Walker hound',
            167: 'English foxhound',
            168: 'redbone',
            169: 'borzoi',
            170: 'Irish wolfhound',
            171: 'Italian greyhound',
            172: 'whippet',
            173: 'Ibizan hound',
            174: 'Norwegian elkhound',
            175: 'otterhound',
            176: 'Saluki',
            177: 'Scottish deerhound',
            178: 'Weimaraner',
            179: 'Staffordshire bullterrier',
            180: 'American Staffordshire terrier',
            181: 'Bedlington terrier',
            182: 'Border terrier',
            183: 'Kerry blue terrier',
            184: 'Irish terrier',
            185: 'Norfolk terrier',
            186: 'Norwich terrier',
            187: 'Yorkshire terrier',
            188: 'wire-haired fox terrier',
            189: 'Lakeland terrier',
            190: 'Sealyham terrier',
            191: 'Airedale',
            192: 'cairn',
            193: 'Australian terrier',
            194: 'Dandie Dinmont',
            195: 'Boston bull',
            196: 'miniature schnauzer',
            197: 'giant schnauzer',
            198: 'standard schnauzer',
            199: 'Scotch terrier',
            200: 'Tibetan terrier',
            201: 'silky terrier',
            202: 'soft-coated wheaten terrier',
            203: 'West Highland white terrier',
            204: 'Lhasa',
            205: 'flat-coated retriever',
            206: 'curly-coated retriever',
            207: 'golden retriever',
            208: 'Labrador retriever',
            209: 'Chesapeake Bay retriever',
            210: 'German short-haired pointer',
            211: 'vizsla',
            212: 'English setter',
            213: 'Irish setter',
            214: 'Gordon setter',
            215: 'Brittany spaniel',
            216: 'clumber',
            217: 'English springer',
            218: 'Welsh springer spaniel',
            219: 'cocker spaniel',
            220: 'Sussex spaniel',
            221: 'Irish water spaniel',
            222: 'kuvasz',
            223: 'schipperke',
            224: 'groenendael',
            225: 'malinois',
            226: 'briard',
            227: 'kelpie',
            228: 'komondor',
            229: 'Old English sheepdog',
            230: 'Shetland sheepdog',
            231: 'collie',
            232: 'Border collie',
            233: 'Bouvier des Flandres',
            234: 'Rottweiler',
            235: 'German shepherd',
            236: 'Doberman',
            237: 'miniature pinscher',
            238: 'Greater Swiss Mountain dog',
            239: 'Bernese mountain dog',
            240: 'Appenzeller',
            241: 'Entlebucher',
            242: 'boxer',
            243: 'bull mastiff',
            244: 'Tibetan mastiff',
            245: 'French bulldog',
            246: 'Great Dane',
            247: 'Saint Bernard',
            248: 'Eskimo dog',
            249: 'malamute',
            250: 'Siberian husky',
            251: 'affenpinscher',
            252: 'basenji',
            253: 'pug',
            254: 'Leonberg',
            255: 'Newfoundland',
            256: 'Great Pyrenees',
            257: 'Samoyed',
            258: 'Pomeranian',
            259: 'chow',
            260: 'keeshond',
            261: 'Brabancon griffon',
            262: 'Pembroke',
            263: 'Cardigan',
            264: 'toy poodle',
            265: 'miniature poodle',
            266: 'standard poodle',
            267: 'Mexican hairless',
            268: 'timber wolf'
        }
        
        self.dog_breed_to_index = {v.lower(): k for k, v in self.imagenet_dog_classes.items()}
        
        logger.info(f"📋 Loaded {len(self.imagenet_dog_classes)} ImageNet dog breeds for classification")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image: Image.Image, top_k: int = 3) -> List[Dict[str, float]]:
        try:
            input_tensor = self.preprocess_image(image)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, min(top_k * 2, 10), dim=1)
            
            results = []
            for i in range(len(top_probs[0])):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                
                if 151 <= idx <= 268 and prob > 0.01:
                    breed_name = self.imagenet_dog_classes.get(idx, "Unknown dog breed")
                    
                    if not any(r['breed'] == breed_name for r in results):
                        results.append({
                            'breed': breed_name,
                            'probability': round(prob * 100, 2)
                        })
                        
                        if len(results) >= top_k:
                            break
            
            if not results:
                logger.warning("No dog breeds detected in the image")
                results = [{
                    'breed': 'No dog detected',
                    'probability': 0.0
                }]
            
            results.sort(key=lambda x: x['probability'], reverse=True)
            
            logger.info(f"🎯 Dog breed prediction completed: {len(results)} breeds identified")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    
    
    

classifier_service = None

def get_classifier_service() -> DogClassifierService:
    global classifier_service
    if classifier_service is None:
        classifier_service = DogClassifierService()
    return classifier_service