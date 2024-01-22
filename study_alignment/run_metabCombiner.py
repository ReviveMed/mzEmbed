
# get the current file path
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path_to_r_script = dir_path + "/metabCombiner_wrapper.R"

input_dir = "/Users/jonaheaton/ReviveMed Dropbox/Jonah Eaton/Codebooks/metaCombiner_example/"
dataset1_path = os.path.join(input_dir, "rcc3/peak_df.txt")
dataset2_path = os.path.join(input_dir, "columbia2711_hilic_pos/peak_df.txt")
output_path = os.path.join(input_dir, "match_ids.csv")

tmp_dir = os.makedirs('tmp', exist_ok=True)
tmp_data1_path = os.path.join('tmp', 'data1.txt')
tmp_data2_path = os.path.join('tmp', 'data2.txt')
tmp_output_path = os.path.join('tmp', 'output.csv')

# copy the input files to the tmp directory
import shutil
shutil.copyfile(dataset1_path, tmp_data1_path)
shutil.copyfile(dataset2_path, tmp_data2_path)


# os.system("Rscript " + path_to_r_script + " " + dataset1_path + " " + dataset2_path + " " + output_path)
os.system("Rscript " + path_to_r_script + " 'tmp/data1.txt' 'tmp/data2.txt' 'tmp/output.csv' FALSE")

# copy the output file to the output directory
shutil.copyfile(tmp_output_path, output_path)

# delete the tmp directory
shutil.rmtree('tmp')

# import rpy2.robjects as robjects
# load the R script
# r = robjects.r
# r.source(path_to_r_script)

# # run the R function
# metaCombiner_wrapper = robjects.globalenv['main']

# # Convert Python arguments to R objects and call the function
# dataset1_path_str = robjects.vectors.StrVector(dataset1_path)
# dataset2_path_str = robjects.vectors.StrVector(dataset2_path)
# output_path_str = robjects.vectors.StrVector(output_path)
# metaCombiner_wrapper(dataset1_path_str, dataset2_path_str, output_path_str)

