#import streamlit as st
#st.write('Hello')
import streamlit as st
import math
import numpy as np
from streamlit import components
from graphviz import Digraph

class Nodes:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob=prob
        self.symbol=symbol
        self.left=left
        self.right=right
        self.code=''

    

def calFreq(the_data):   #calculates frequency of letters
    the_symbols=dict()      
    for i in the_data:
        if the_symbols.get(i)==None:
            the_symbols[i]=1
        else:
            the_symbols[i]+=1
    return the_symbols

codes = dict()   #store codes for each letter

def calCodes(node, val=''):
    newval=val+str(node.code)
    if node.left:
        calCodes(node.left, newval)
    if node.right:
        calCodes(node.right, newval)
    else:
        codes[node.symbol]=newval
    return codes

def encodedOutput(the_data, coding):
    l=[]
    for i in the_data:
        l.append(coding[i])
    
    ans=''.join([str(i) for i in l])
    return ans

def TotalGain(the_data, coding, symbol_frequencies):
    n = len(symbol_frequencies)
    a = np.log2(n)
    bits = int(np.ceil(a))
    befComp = len(the_data) * bits
    afComp = 0
    the_symbols = coding.keys()
    for symbol in the_symbols:
        the_count = the_data.count(symbol)
        afComp+= the_count * len(coding[symbol])
    return befComp, afComp

def calculateEntropy(probabilities):
    entropy = 0
    sum=0
    for probability in probabilities:
        sum+=probability
    for probability in probabilities:
        entropy += (probability/sum) * math.log2(sum/probability)
    entropy=round(entropy,2)
    return entropy

def calculateAverageLength(coding, probabilities):
    averageLength = 0
    sumq = 0
    for probability in probabilities.values():
        sumq += float(probability)
    for symbol, probability in probabilities.items():
        averageLength += (float(probability)/sumq) * len(coding[symbol])
    averageLength = round(averageLength, 2)
    return averageLength


def HuffmanEncoding(the_data):
    symbolWithProbs = calFreq(the_data)
    the_symbols = symbolWithProbs.keys()
    the_prob = symbolWithProbs.values()

    the_nodes = []

    for symbol in the_symbols:
        the_nodes.append(Nodes(symbolWithProbs.get(symbol), symbol))

    while len(the_nodes) > 1:
        the_nodes = sorted(the_nodes, key=lambda x: x.prob)
        right = the_nodes[0]
        left = the_nodes[1]

        left.code = 0
        right.code = 1

        newNode = Nodes(left.prob + right.prob, left.symbol + right.symbol, left, right)

        the_nodes.remove(left)
        the_nodes.remove(right)
        the_nodes.append(newNode)

    huffmanEncoding = calCodes(the_nodes[0])
    befComp, afComp = TotalGain(the_data, huffmanEncoding, symbolWithProbs)
    entropy = calculateEntropy(the_prob)
    averageLength = calculateAverageLength(huffmanEncoding, symbolWithProbs)
    output = encodedOutput(the_data, huffmanEncoding)


    
# Generate the modified encoded output with horizontal curly brackets below each symbol
    modified_output = ""
    for symbol in the_data:
        bits = huffmanEncoding[symbol]
        modified_output += f"\\underbrace{{\\Large {bits}}}_{{{symbol}}} \\quad "



    return modified_output, the_nodes[0], befComp, afComp, entropy, averageLength

def HuffmanDecoding(encoded_string, huffman_tree):
    decoded_string = ""
    all_uniq_chars_in_encoded_string = set(encoded_string)
    # check set has any char other than 0 and 1
    diff = all_uniq_chars_in_encoded_string - {'0', '1'}
    if diff:
        raise ValueError(f"Invalid Huffman code. Characters other than 0 and 1 found\n {diff}")
    current_node = huffman_tree
    for bit in encoded_string:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.left is None and current_node.right is None:
            decoded_string += current_node.symbol
            current_node = huffman_tree

    if current_node != huffman_tree:
        #during raise value error, it will show the bit index where the error occured
        bit_index = len(encoded_string) - len(bit)
        raise ValueError(f"Incorrect Huffman code. Decoding stopped at bit {bit_index}")
        

    

    return decoded_string


uniform = dict()

def print_symbol_frequencies(symbol_frequencies):
    n = len(symbol_frequencies)
    a = np.log2(n)
    bits = int(np.ceil(a))
    
    for index, symbol in enumerate(symbol_frequencies):
        indInt = int(index)
        co = bin(indInt)[2:].zfill(bits)
        uniform[symbol] = co

    table_data = [["Symbol", "Frequency", "Uniform Code Length", "Huffman Code"]]
    for symbol, frequency in symbol_frequencies.items():
        code = codes.get(symbol, "")
        table_data.append([symbol, frequency, uniform.get(symbol), code])

    st.table(table_data)

def print_huffman_tree(node):
    dot = Digraph()
    dot.node('root', label=f"{node.symbol}\n{node.prob}", shape='circle', style='filled', color='white', fillcolor='red')
    create_graph(node, dot, 'root')
    
    # Set graph attributes
    dot.graph_attr.update(bgcolor='None', size='10,10')
    dot.node_attr.update(shape='circle', style='filled', color='white')
    dot.edge_attr.update(color='white', fontcolor='white')

    # Render the graph
    dot.format = 'png'
    dot.render('huffman_tree', view=False)

    st.image('huffman_tree.png')

def create_graph(node, dot, parent=None):
    if node.left:
        dot.node(str(node.left), label=f"{node.left.symbol}\n{node.left.prob}", style='filled', fillcolor='#00CCFF')
        if parent:
            dot.edge(str(parent), str(node.left), label='0', color='#FF0000')
        create_graph(node.left, dot, node.left)

    if node.right:
        dot.node(str(node.right), label=f"{node.right.symbol}\n{node.right.prob}", style='filled', fillcolor='#00CCFF')
        if parent:
            dot.edge(str(parent), str(node.right), label='1', color='#00FF00')
        create_graph(node.right, dot, node.right)
        
def main():
    st.markdown("<h1 style='text-align: center; color: #457B9D;'>Huffman Coding</h1>", unsafe_allow_html=True)
    the_data = st.text_input("Enter the data:", "sustainibilitylab")
    encoding, the_tree, befComp, afComp, entropy, averageLength = HuffmanEncoding(the_data)
    st.write("Encoded Output: ")
    st.latex(encoding)

    st.markdown("<h2 style='font-size: 24px;text-align: center; color: #457B9D;'>Huffman Tree:</h2>", unsafe_allow_html=True)
    print_huffman_tree(the_tree)

    st.markdown("<h2 style='font-size: 24px; text-align: center;color: #457B9D;'>Symbols, Frequencies and Codes:</h2>", unsafe_allow_html=True)
    symbol_frequencies = calFreq(the_data)
    print_symbol_frequencies(symbol_frequencies)
    

    st.markdown("<h2 style='font-size: 24px;text-align: center; color: #457B9D;'>Encoding Properties:</h2>", unsafe_allow_html=True)
    
    st.write("Before Compression with uniform code length(no. of bits): ", befComp)
    st.write("After Compression woth huffman code (no. of bits): ", afComp)
    st.write("Entropy: ", entropy)
    st.write("Average Length: ", averageLength)
    
    st.markdown("<h2 style='font-size: 24px;text-align: center; color: #457B9D;'>Huffman Decoding:</h2>", unsafe_allow_html=True)
    encoded_input = st.text_input("Enter the encoded string:")
    decoded_output = HuffmanDecoding(encoded_input, the_tree)
    st.write("Decoded Output:", decoded_output)



hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



if __name__ == '__main__':
    main()
