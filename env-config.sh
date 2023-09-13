# Install dependencies in anconda environment

source activate FO4-Terminal

conda install pillow

conda install -c menpo opencv3

pip install pytesseract

# Export Environment in .yml file
conda env export > FO4-Terminal.yml

