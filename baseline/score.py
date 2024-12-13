import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_base_name(filename, over_dark=None):

    # Split filename and extension
    name, ext = os.path.splitext(filename)
    # print(f"name: {name}, ext: {ext}")
    
    # If the filename ends with '_0', map it to '_P1.5'
    if name.endswith('_0'):
        if over_dark == "over":
            base_name = name[:-2] + '_P1.5' + ext  # Remove '_0', add '_P1.5' and extension
        elif over_dark == "dark":
            base_name = name[:-2] + '_N1.5' + ext
        else:
            print("Error: over_dark should be either 'over' or 'dark'")
            exit()
    else:
        base_name = name + ext
    
    # print(f"base_name: {base_name}")
    return base_name

def calculate_metrics(groundtruth_folder, folders_to_compare, output_file="metrics_results.csv", result_name = None, over_dark = None):

    results = {}
    
    # Create mapping for ground truth files
    gt_files = {}
    for f in os.listdir(groundtruth_folder):
        base_name = get_base_name(f, over_dark)
        gt_files[base_name] = f
        # print(f"base_name: {base_name}, f: {f}")
    
    for folder in folders_to_compare:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue
        
        print("=============================================================================")
        print(f"Processing folder: {folder}")
        for filename in os.listdir(folder):
            base_name = get_base_name(filename, over_dark)
            
            # Find matching ground truth file
            if base_name not in gt_files:
                print(f"No matching ground truth file for: {filename}")
                continue
                
            gt_filename = gt_files[base_name]
            gt_path = os.path.join(groundtruth_folder, gt_filename)
            compare_path = os.path.join(folder, filename)

            if not os.path.exists(gt_path):
                print(f"Ground truth file not found: {gt_path}")
                continue

            if not os.path.exists(compare_path):
                print(f"Comparison file not found: {compare_path}")
                continue

            # Read images
            gt_img = cv2.imread(gt_path)
            compare_img = cv2.imread(compare_path)
            print(f"Comparing: {gt_filename} with {filename}")

            if gt_img is None or compare_img is None:
                print(f"Error reading images: {filename}")
                continue

            # Ensure images are the same size
            if gt_img.shape != compare_img.shape:
                print(f"Image size mismatch: {filename}. Resizing comparison image.")
                compare_img = cv2.resize(compare_img, (gt_img.shape[1], gt_img.shape[0]))

            # Convert to grayscale if images are not multichannel
            is_multichannel = gt_img.ndim == 3

            try:
                # Compute PSNR
                p = psnr(gt_img, compare_img, data_range=gt_img.max() - gt_img.min())

                # Compute SSIM
                s = ssim(gt_img, compare_img, channel_axis=-1 if is_multichannel else None)

                if filename not in results:
                    results[filename] = {"File": filename}
                results[filename][folder] = {"PSNR": p, "SSIM": s}
                
            except Exception as e:
                print(f"Error calculating metrics for {filename}: {str(e)}")
                continue

    # Save results to a CSV file
    import pandas as pd
    final_results = []
    # print(f"{results=}")
    for filename, metrics in results.items():
        row = {"File": filename[0:5]}
        
        # 分別收集所有資料夾的 PSNR 和 SSIM
        for folder in folders_to_compare:
            # 如果這個資料夾有該檔案的度量結果
            if folder in metrics:
                # 將所有 PSNR 放在一起
                # folder_name = folder.split('/')[-1]
                folder_name = result_name[folders_to_compare.index(folder)]
                # print(f"{folder_name=}")
                row[f"PSNR_{folder_name}"] = metrics[folder]["PSNR"]
        
        for folder in folders_to_compare:
            # 如果這個資料夾有該檔案的度量結果
            if folder in metrics:
                # 將所有 SSIM 放在一起
                # folder_name = folder.split('/')[-1]
                folder_name = result_name[folders_to_compare.index(folder)]
                # print(f"{folder_name=}")
                row[f"SSIM_{folder_name}"] = metrics[folder]["SSIM"]
                
        final_results.append(row)

    df = pd.DataFrame(final_results)

    # 重新排序列，讓所有 PSNR 和 SSIM 分別集中
    psnr_columns = [col for col in df.columns if 'PSNR' in col]
    ssim_columns = [col for col in df.columns if 'SSIM' in col]
    # 設定最終的列順序：File, 所有PSNR, 所有SSIM
    column_order = ['File'] + psnr_columns + ssim_columns

    # 按照新的順序重排列
    df = df[column_order]
    df.to_csv(output_file, index=False)
    print(f"Metrics saved to {output_file}")


def main():
    # groundtruth_folder = "data/ground_truth/ground_truth_over"
    # folders_to_compare = [
    #     "data/result/histogram/data_over",
    #     "data/result/zero_dce/data_over",
    # ]
    # groundtruth_folder = "data/ground_truth/ground_truth_dark"
    # folders_to_compare = [
    #     "data/result/histogram/data_dark",
    #     "data/result/zero_dce/data_dark",
    # ]
    
    groundtruth_folder = "data/ground_truth/ground_truth_dark"
    folders_to_compare = [
        # "data/result/laplacian_bad/underexposed_bad",
        # "data/result/laplacian_bad_fuse/underexposed_bad_fuse",
        # "data/result/laplacian_best/underexposed_best",
        # "data/result/laplacian_best_fuse/underexposed_best_fuse",
        "",
    ]
    
    result_name = ["bad_backward"]
    # create folder if not exist
    if not os.path.exists("data/metrics"):
        os.makedirs("data/metrics")
    # output_file = "data/metrics/metrics_baseline_over.csv"
    # output_file = "data/metrics/metrics_baseline_dark.csv"
    output_file = "data/metrics/metrics_our_dark.csv"
    over_dark = "dark"
    calculate_metrics(groundtruth_folder, folders_to_compare, output_file, result_name, over_dark)

    
if __name__ == "__main__":
    main()