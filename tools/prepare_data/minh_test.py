from glob import glob
import pandas as pd
import subprocess

def create_vox2_train_csv():
    files = glob("/media/volume/AudioUnlearnData1/sslsv/data/voxceleb2/*/*/*.wav")
    print(len(files))
    files.sort()

    df = pd.DataFrame({"File": files, "Speaker": [f.split("/")[-3] for f in files]})

    df.to_csv("data/voxceleb2_train.csv", index=False)


TRIALS = [
    (
        "voxceleb1_test_O",
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt",
    ),
    (
        "voxceleb1_test_H",
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt",
    ),
    (
        "voxceleb1_test_E",
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt",
    ),
    (
        "voxsrc2021_val",
        "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/voxsrc2021_val.txt",
    ),
]

def create_vox_trials():
    for filename, url in TRIALS:
        status = subprocess.call("wget %s -O %s" % (url, filename), shell=True)
        if status != 0:
            raise Exception("Download of %s failed" % filename)

    VOX_TRIALS = [
        "voxceleb1_test_O",
        "voxceleb1_test_E",
        "voxceleb1_test_H",
        "voxsrc2021_val",
    ]

    # Add voxceleb1 prefix to trial files
    for trial in VOX_TRIALS:
        res = []
        with open(trial) as f:
            for line in f.readlines():
                target, a, b = line.split()

                line_ = f"{target} voxceleb1/{a} voxceleb1/{b}"

                res.append(line_)

        with open(trial, "w") as f:
            f.write("\n".join(res))

if __name__ == "__main__":
    create_vox2_train_csv()
    create_vox_trials()