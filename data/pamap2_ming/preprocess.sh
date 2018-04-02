#please put all data here

train_files='subject101_processed.dat subject102_processed.dat subject103_processed.dat subject104_processed.dat subject107_processed.dat subject108_processed.dat subject109_processed.dat'
val_file="subject105_processed.dat"
test_file="subject106_processed.dat"

rm -f train.dat
rm -f test.dat
rm -f validate.dat

for f in $train_files; do
	echo "Merging " $f
	cat $f >> train.dat
done

cat $val_file >> validate.dat
cat $test_file >> test.dat

