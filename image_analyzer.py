# [file name]: image_analyzer.py
# [file content begin]
import torch
import logging
from transformers import pipeline
from gate_detector import LogicGateDetector
from langchain_ollama.llms import OllamaLLM
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerilogImageAnalyzer:
    def __init__(self):
        try:
            self.device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'cuda' if self.device == 0 else 'cpu'}")

            logger.info("Loading image-to-text model...")
            self.caption_model = pipeline(
                "image-text-to-text",
                model="Salesforce/blip-image-captioning-large",
                device=self.device
            )
            logger.info("Image-to-text model loaded.")

            logger.info("Initializing logic gate detector...")
            self.gate_detector = LogicGateDetector()
            logger.info("Logic gate detector initialized.")
            
            logger.info("Initializing text enhancement model...")
            self.text_model = OllamaLLM(
                model="codellama:7b-instruct",
                temperature=0.1,
                top_k=40,
                top_p=0.9,
                num_ctx=4096
            )
            logger.info("Text enhancement model loaded.")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def analyze_image(self, image_path):
        try:
            logger.info(f"Starting image analysis for: {image_path}")

            # 1. Get caption using BLIP model
            logger.info("Generating image caption...")
            try:
                caption_result = self.caption_model(image_path)
                caption = caption_result[0]['generated_text'] if caption_result else "No caption generated"
                
                # Enhance caption with Codellama
                enhanced_prompt = f"Refine this circuit description to be more technical and precise: {caption}"
                enhanced_caption = self.text_model.invoke(enhanced_prompt)
                caption = enhanced_caption.content if hasattr(enhanced_caption, 'content') else enhanced_caption
            except Exception as caption_error:
                logger.warning(f"Caption generation failed: {str(caption_error)}")
                caption = "No caption available"

            logger.info(f"Caption: {caption}")

            # 2. Detect gates and connections using LogicGateDetector
            logger.info("Detecting logic gates and connections...")
            gates, connections = self.gate_detector.detect_circuit(image_path)

            # Estimate bit-width based on detected components and caption
            bit_width = self._estimate_bit_width(gates, caption)

            logger.info("Analysis complete.")
            return {
                "caption": caption,
                "gates": gates,
                "connections": connections,
                "bit_width": bit_width,
                "description": caption
            }

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "error": str(e),
                "bit_width": 1,
                "description": "Error occurred during analysis"
            }

    def _estimate_bit_width(self, gates, caption=""):
        """Estimate bit-width based on detected gates and caption"""
        if not gates:
            return 1
        
        # Look for N-bit components in gate types
        for gate in gates:
            if 'N-bit' in gate['type'] or 'n-bit' in gate['type']:
                return 8  # Default to 8-bit if N-bit components found
        
        # Look for bit-width in caption
        bit_match = re.search(r'(\d+)-bit', caption, re.IGNORECASE)
        if bit_match:
            return int(bit_match.group(1))
        
        return 1  # Default to 1-bit for simple gates

    def analyze_text(self, text_description):
        """Analyze text description to extract components and estimate bit-width"""
        try:
            logger.info(f"Analyzing text description: {text_description}")
            
            # Extract bit-width from text
            bit_width = 1
            bit_match = re.search(r'(\d+)-bit', text_description, re.IGNORECASE)
            if bit_match:
                bit_width = int(bit_match.group(1))
            
            # Extract potential components from text
            components = []
            component_keywords = {
                "AND": ["and", "and gate"],
                "OR": ["or", "or gate"],
                "NOT": ["not", "inverter", "not gate"],
                "NAND": ["nand", "nand gate"],
                "NOR": ["nor", "nor gate"],
                "XOR": ["xor", "exor", "xor gate"],
                "XNOR": ["xnor", "exnor", "xnor gate"],
                "MUX": ["mux", "multiplexer"],
                "DEMUX": ["demux", "demultiplexer"],
                "ENCODER": ["encoder"],
                "DECODER": ["decoder"],
                "FLIP-FLOP": ["flip-flop", "ff", "dff", "jkff", "tff"],
                "REGISTER": ["register"],
                "COUNTER": ["counter"],
                "ADDER": ["adder", "half adder", "full adder"],
                "SUBTRACTOR": ["subtractor"],
                "COMPARATOR": ["comparator"],
                "BUFFER": ["buffer", "tri-state"],
                "DAC": ["dac", "digital-to-analog"],
                "ADC": ["adc", "analog-to-digital"],
                "RAM": ["ram", "memory"],
                "ROM": ["rom", "read-only memory"]
            }
            
            text_lower = text_description.lower()
            for comp_type, keywords in component_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        components.append({
                            "type": comp_type,
                            "count": text_lower.count(keyword)
                        })
                        break
            
            return {
                "description": text_description,
                "components": components,
                "bit_width": bit_width
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return {
                "error": str(e),
                "bit_width": 1,
                "description": text_description
            }
# [file content end]