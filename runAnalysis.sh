mkdir outputs

python3 boxplot.py
python3 umapProj.py
python3 method_comp.py | tee outputs/method_comp_results.txt
python3 perCancerGenes.py | tee outputs/perCancerGenes_results.txt
echo Done!
