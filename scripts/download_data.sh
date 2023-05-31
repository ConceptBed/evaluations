mkdir datasets

wget https://www.dropbox.com/s/jiii17xu2i5trwm/imagenet_filtered_v1_0.zip?dl=0 -P ./datasets/
mv ./datasets/imagenet_filtered_v1_0.zip?dl=0 ./datasets/imagenet_filtered_v1_0.zip
unzip ./datasets/imagenet_filtered_v1_0.zip -d ./datasets/imagenet_subsection/

wget https://www.dropbox.com/s/294xzd41clcwn72/compositions_v1_0.zip?dl=0 -P ./datasets/
mv ./datasets/compositions_v1_0.zip?dl=0 ./datasets/compositions_v1_0.zip
unzip ./datasets/compositions_v1_0.zip -d ./datasets/compositions/
