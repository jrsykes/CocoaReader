import os
import shutil

base = '/local/scratch/jrs596/dat/ResNetFung50+_images_organised_subset_TrainValCombined/'
dirs =  os.listdir(base)

for i in dirs:
	if i == 'val':
		classes = os.listdir(base + i)
		for j in classes:
			images = os.listdir(base + i + '/' + j)
			for k in images:
				origin = base + i + '/' + j + '/' + k
				target = base + 'train' + '/' + j + '/' + k
				os.rename(origin, target)
				
#full = os.listdir('/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train')#

#subset_list = ['AauberginesDiseased', 'BananasDiseased', 'CabbagesHealthy', 'CherryHealthy', 'CucumbersHealthy', 'LettuceHealthy', 'MangosHealthy', 'OnionsHealthy', 'PeasHealthy', 'RiceDiseased', 'StrawberriesDiseased', 'SunflowerHealthy', 'TobaccoHealthy', 'WatermelonsHealthy',
#	'AauberginesHealthy', 'BananasHealthy', 'CassavaDiseased', 'CocoaDiseased', 'GarlicHealthy', 'MaizeDiseased', 'OlivesDiseased', 'OrangesDiseased', 'PotatoesDiseased', 'SoybeansHealthy', 'StrawberriesHealthy', 'Sweet_potatoesHealthy', 'TomatoesDiseased', 'WheatHealthy',
#	'ApplesDiseased', 'BarleyDiseased', 'CassavaHealthy', 'CocoaHealthy', 'GrapesDiseased', 'MaizeHealthy', 'OlivesHealthy', 'OrangesHealthy', 'PotatoesHealthy', 'SpinachDiseased', 'SugarcaneDiseased', 'TeaDiseased', 'TomatoesHealthy',
#	'ApplesHealthy', 'CabbagesDiseased', 'CherryDiseased', 'CucumbersDiseased', 'GrapesHealthy', 'MangosDiseased', 'OnionsDiseased', 'PeachesHealthy', 'RapeseedHealthy', 'SpinachHealthy', 'SugarcaneHealthy', 'TeaHealthy'] #
#
#

#for i in os.listdir('/local/scratch/jrs596/dat/ResNetFung50+_images_organised_subset/test'):
#	if i not in subset_list:
#		shutil.rmtree('/local/scratch/jrs596/dat/ResNetFung50+_images_organised_subset/test/' + i)

