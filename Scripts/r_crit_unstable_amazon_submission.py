import os

file = "r_crit_unstable_amazon_submit.txt"
num_lines = sum(1 for line in open(file))



with open(file) as fp:
    for cnt in range(0, num_lines):
        line = fp.readline()
        print(line)

        #iniate job script
        with open("job_submit.sh", "w+") as fh:
            fh.writelines("#!/bin/bash\n\n")

            #specifications of the job that should be submitted
            fh.writelines("#SBATCH --qos=short\n")
            fh.writelines("#SBATCH --job-name=tipping\n")
            fh.writelines("#SBATCH --account=lenaschm\n\n") #workinggroup

            fh.writelines("#SBATCH --workdir=/home/lenaschm/lena\n") #ändern zu meinem wd
            fh.writelines("#SBATCH --output=outfile-%j.txt\n")
            fh.writelines("#SBATCH --error=error-%j.txt\n")
            fh.writelines("#SBATCH --nodes=1\n")
            fh.writelines("#SBATCH --ntasks-per-node=1\n")
            #fh.writelines("#SBATCH --exclusive\n")
            fh.writelines("#SBATCH --time=0-23:50:00\n\n")

            fh.writelines("module load anaconda/5.0.0_py3\n") # ändern
            fh.writelines("source activate tipping\n\n") #my environment

            #job to be submitted
            fh.writelines("{}".format(line))
            fh.close()

        os.system("sbatch {}".format("job_submit.sh"))

