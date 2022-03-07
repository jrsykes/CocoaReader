import shutil
import os



def resample_data(temp_dir, k, split):
    try:
        shutil.rmtree(temp_dir + 'train')
        shutil.rmtree(temp_dir + 'val')
    except:
        pass

    for i in ['FPR', 'WBD', 'Healthy', 'BPR']:
        image_list = os.listdir(data_dir + '/' + i)
        #random.shuffle(image_list)
        for j in ['train', 'val']:
            os.makedirs(temp_dir + j + '/' + i, exist_ok = True)
            if j == 'val':

                sample = image_list[:int(len(image_list)*split)]  #fist 10, 20, 30
                if split > 0.1:
                    sample = sample[int(len(image_list)*(split-0.1)):]
                for s in sample:
                    source = data_dir + '/' + i + '/' + s
                    dest = temp_dir + j + '/' + i + '/' + s
                    shutil.copy(source, dest)

                for im in image_list:
                    if im not in sample:
                        source = data_dir + '/' + i + '/' + im
                        dest = temp_dir + 'train/' + i + '/' + im
                        shutil.copy(source, dest)


data_dir = "/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal"
split = 0.1

import time

for k in range(10):

    previous_list = ['image0.jpeg', '_40311379_pod303_aberystwyth.jpg', 'broom200-e90129e040a203b164c089f8017aa0ae43cd5d48-s800-c85.jpg', '0415-diseased-cacao-812x1200.jpg', 'images326.jpg', 'Cacao_Fig11.jpg', 'images674.jpg', 'image12.jpeg', 'Selected-hosts-of-Moniliophthora-perniciosa-a-A-green-terminal-broom-on-cupuassu_Q320.jpg', 'image14.jpeg', 'images267.jpg', 'img_0909.jpg', 'Crinipellis_perniciosa_mushroom.jpg', 'ppa12204-fig-0001-m.jpg', 'images106.jpg', 'SS2152402.jpg', 'A-cacao-tree-affected-by-witchesE28099-broom.jpg', 'images671.jpg', '1-s2.0-S1878614620301276-gr2.jpg', 'images290.jpg', 'Witches252520Broom252520disease252520attacking252520a252520cocoa252520tree252520in252520Quevedo252C252520Los252520Rios.jpg', 'image17.jpeg', '9860.jpg', 'witch_broom_sJPG.jpg', 'images176.jpg', 'witches20broom.jpg', 'images87.jpg', 'images183.jpg', 'images361.jpg', 'img_0911.jpg', 'image18.jpeg', 'images316.jpg', 'ibroom.jpg', 'Cacao_Fig12.jpg', 'The-broom-shaped-stalks-of-an-infected-tree.jpg', 'images502.jpg', '3-s2.0-B9780080473789500178-f11-0163-9780080473789.jpg', '9859.jpg', 'image1.jpeg', 'image9.jpeg', 'images298.jpg', 'dsc00937.jpg', 'images381.jpg', 'images261.jpg', '0415-pink-mushrooms296x460.jpg', 'images187.jpg', 'F1.large.jpg', 'FPR-WB_tog5_sJPG.jpg', 'images278.jpg', 'image.jpeg', 'images447.jpg', 'images339.jpg', 'images450.jpg', '1317028.jpg', 'images250.jpg', 'image15.jpeg', 'witches-broom.jpg', 'image13.jpeg']


    resample_data('/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal_epoch_randomisation/', k, split)

    split += 0.1
    for i in os.listdir('/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal_epoch_randomisation/train/WBD/'):
        count = 0
        if i in previous_list:
            count += 1
            print(count)
            print(len(os.listdir('/local/scratch/jrs596/dat/compiled_cocoa_images/CrossVal_epoch_randomisation/train/WBD/')))
        previous_list.append(i)
