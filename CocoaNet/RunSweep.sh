n=13; command="sbatch /jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts/CocoaReader/CocoaNet/Run_J2.sh"; for i in $(seq 1 $n); do echo "Running command iteration $i"; eval "$command"; done
