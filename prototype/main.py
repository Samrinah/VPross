from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import create_retriever
from image_analyzer import VerilogImageAnalyzer
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_components():
    try:
        logger.info("Initializing vector retriever...")
        retriever = create_retriever()
        
        logger.info("Initializing LLM...")
        model = OllamaLLM(
            model="codellama:7b-instruct",
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            num_ctx=4096
        )
        logger.info("Using model: codellama:7b-instruct")

        logger.info("Initializing image analyzer...")
        image_analyzer = VerilogImageAnalyzer()
        
        return retriever, model, image_analyzer
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

def generate_verilog_code(chain, analysis, retriever):
    try:
        # Handle both image-based and text-based analysis
        if "gates" in analysis:
            # Image analysis format
            components = "\n".join([f"- {gate['type']} ({gate['id']})" for gate in analysis["gates"]])
            connections = "\n".join([f"{conn['source']} -> {conn['target']} (input {conn['input_num']})" for conn in analysis["connections"]])
            
            if analysis["gates"]:
                example_query = f"{analysis['bit_width']}-bit {analysis['gates'][0]['type']} gate implementation"
                context = retriever.invoke(example_query)
                examples = "\n\n".join([doc.page_content for doc in context])
            else:
                examples = "No gates detected."
        else:
            # Text analysis format
            components = "\n".join([f"- {comp['type']} (x{comp['count']})" for comp in analysis.get("components", [])])
            connections = "Connection details not available from text description"
            example_query = f"{analysis['bit_width']}-bit circuit implementation"
            context = retriever.invoke(example_query)
            examples = "\n\n".join([doc.page_content for doc in context])

        verilog = chain.invoke({
            "components": components,
            "connections": connections,
            "bit_width": analysis["bit_width"],
            "description": analysis["description"],
            "examples": examples
        })
        return verilog.content if hasattr(verilog, 'content') else verilog
    except Exception as e:
        logger.error(f"Verilog generation failed: {str(e)}")
        return f"Error generating Verilog: {str(e)}"

def main():
    try:
        retriever, model, image_analyzer = initialize_components()

        template = """You are a Verilog expert. Generate complete, synthesizable code based on:

[COMPONENT ANALYSIS]
Detected components: {components}
Connection pattern: {connections}
Estimated bit-width: {bit_width}

[DESCRIPTION]
{description}

[RELEVANT EXAMPLES]
{examples}

[INSTRUCTIONS]
1. Use {bit_width}-bit implementations
2. Maintain consistent bit-widths
3. Include module declarations
4. Connect components as analyzed
5. Add detailed comments"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        print("\n" + "="*60)
        print("Verilog Code Generator")
        print("="*60)

        user_input = input("Enter circuit image path or text description: ").strip()

        analysis = None
        start_time = time.time()

        # Auto-detect: image path vs text
        if os.path.exists(user_input) and user_input.lower().endswith(('.png', '.jpg', '.jpeg')):
            print("\nAnalyzing circuit image...")
            analysis = image_analyzer.analyze_image(user_input)
        else:
            print("\nAnalyzing text description...")
            analysis = image_analyzer.analyze_text(user_input)

        analysis_time = time.time() - start_time

        if "error" in analysis:
            print(f"Analysis Error: {analysis['error']}")
            return

        # Print summary
        print(f"\nAnalysis completed in {analysis_time:.2f} seconds")
        if "gates" in analysis:
            print(f"Detected {len(analysis['gates'])} gates")
        else:
            print(f"Detected {len(analysis.get('components', []))} component types")
        print(f"Estimated bit-width: {analysis['bit_width']}")
        print(f"\nCircuit Description: {analysis['description']}")

        # Generate Verilog code
        verilog = generate_verilog_code(chain, analysis, retriever)
        print("\nGenerated Verilog Code:")
        print("="*60)
        print(verilog)
        print("="*60)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("A fatal error occurred. Check logs for details.")

if __name__ == "__main__":
    main()
