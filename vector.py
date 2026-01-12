# [file name]: vector.py
# [file content begin]
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
import pandas as pd
import os
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_retriever(csv_path="dataset.csv", persist_dir="./chroma_verilog_db", force_reload=False):
    try:
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Using the same model family for consistency
        embeddings = OllamaEmbeddings(model="codellama:7b-instruct")
        
        if force_reload or not os.path.exists(persist_dir):
            logger.info("Creating new vector store")
            documents = []
            for _, row in df.iterrows():
                bit_match = re.search(r'(\d+)-bit', row["Component"], re.IGNORECASE)
                bit_width = bit_match.group(1) + "-bit" if bit_match else "1-bit"
                component_type = _determine_component_type(row["Component"])
                
                doc = Document(
                    page_content=(
                        f"Component: {row['Component']}\n"
                        f"Type: {component_type}\n"
                        f"Bit Width: {bit_width}\n"
                        f"Verilog Code:\n{row['Verilog Code']}\n"
                        f"Description: {_generate_component_description(row['Component'], row['Verilog Code'])}"
                    ),
                    metadata={
                        "component": row["Component"],
                        "type": component_type,
                        "bit_width": bit_width,
                        "code": row["Verilog Code"],
                        "description": _generate_component_description(row["Component"], row['Verilog Code'])
                    }
                )
                documents.append(doc)
            
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name="verilog_components"
            )
            logger.info(f"Created new vector store at {persist_dir}")
        else:
            logger.info(f"Loading existing vector store from {persist_dir}")
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name="verilog_components"
            )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        return retriever

    except Exception as e:
        logger.error(f"Failed to create retriever: {str(e)}")
        raise

def _determine_component_type(component_name):
    component_name = component_name.lower()
    gate_types = ["and", "or", "nand", "nor", "xor", "xnor", "not", "buffer"]
    if any(gate in component_name for gate in gate_types):
        return "gate"
    combinational = ["mux", "demux", "encoder", "decoder", "multiplexer", "demultiplexer"]
    if any(comb in component_name for comb in combinational):
        return "combinational"
    sequential = ["flip", "flop", "latch", "register", "counter", "fsm", "state machine"]
    if any(seq in component_name for seq in sequential):
        return "sequential"
    arithmetic = ["adder", "subtractor", "multiplier", "divider", "alu", "comparator"]
    if any(arith in component_name for arith in arithmetic):
        return "arithmetic"
    memory = ["ram", "rom", "memory", "fifo", "stack"]
    if any(mem in component_name for mem in memory):
        return "memory"
    converter = ["dac", "adc", "gray", "binary"]
    if any(conv in component_name for conv in converter):
        return "converter"
    return "other"

def _generate_component_description(component_name, verilog_code):
    """Generate a descriptive text for the component to improve retrieval"""
    component_name = component_name.lower()
    
    if "and" in component_name:
        return "Logic AND gate that performs logical conjunction operation"
    elif "or" in component_name:
        return "Logic OR gate that performs logical disjunction operation"
    elif "not" in component_name:
        return "Logic NOT gate (inverter) that performs logical negation"
    elif "nand" in component_name:
        return "Logic NAND gate that performs negated logical conjunction"
    elif "nor" in component_name:
        return "Logic NOR gate that performs negated logical disjunction"
    elif "xor" in component_name or "exor" in component_name:
        return "Logic XOR gate that performs exclusive OR operation"
    elif "xnor" in component_name or "exnor" in component_name:
        return "Logic XNOR gate that performs exclusive NOR operation"
    elif "mux" in component_name or "multiplexer" in component_name:
        return "Multiplexer that selects one of many input signals"
    elif "demux" in component_name or "demultiplexer" in component_name:
        return "Demultiplexer that routes a single input to one of many outputs"
    elif "encoder" in component_name:
        return "Encoder that converts multiple inputs into a coded output"
    elif "decoder" in component_name:
        return "Decoder that converts coded inputs into multiple outputs"
    elif "flip" in component_name or "flop" in component_name:
        return "Flip-flop that stores a single bit of data"
    elif "register" in component_name:
        return "Register that stores multiple bits of data"
    elif "counter" in component_name:
        return "Counter that increments or decrements a value"
    elif "adder" in component_name:
        return "Adder that performs arithmetic addition"
    elif "subtractor" in component_name:
        return "Subtractor that performs arithmetic subtraction"
    elif "comparator" in component_name:
        return "Comparator that compares two values"
    elif "buffer" in component_name:
        return "Buffer that amplifies or isolates signals"
    elif "dac" in component_name:
        return "Digital-to-Analog Converter that converts digital signals to analog"
    elif "adc" in component_name:
        return "Analog-to-Digital Converter that converts analog signals to digital"
    elif "ram" in component_name:
        return "Random Access Memory for data storage"
    elif "rom" in component_name:
        return "Read-Only Memory for permanent data storage"
    elif "gray" in component_name:
        return "Gray code converter for error reduction in digital systems"
    else:
        return f"Verilog implementation of {component_name}"
# [file content end]