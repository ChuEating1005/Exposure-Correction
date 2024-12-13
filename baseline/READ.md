
run under baseline folder

python baseline.py
    能夠跑 zero_dce, gamma (用arg調參數), histogram
    input: data/test_data
    output: data/result

python score.py
    能夠算 ground_truth 與 folder 的 PSNR 以及 SSIM
    input, output 都是在main那邊自訂

    input_ground_truth: data/ground_truth/ground_truth_over, data/ground_truth/ground_truth_dark
    input_compare: 自訂
    output: data/metrics/{name}.csv