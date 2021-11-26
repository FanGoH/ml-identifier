import json
import matplotlib.pyplot as plt


def load_plot(propertyName, graphTitle):
  fig, ax = plt.subplots()

  f = open('./plottingProject/json/Modelo-90acc.json')
  data = json.load(f)
  ax.plot(data['epochs'][::200], data[propertyName][::200], label=f'Modelo-1')
  for i in range(3, 10):
    f = open(f'./plottingProject/json/Modelo{i}.json')

    data = json.load(f)

    ax.plot(data['epochs'], data[propertyName], label=f'Modelo-{i-2}')


 
  ax.legend(loc=2, bbox_to_anchor=(0.80, 1.0))
  plt.title(graphTitle)

  plt.show()
  

if __name__=='__main__':
  
  load_plot('train_loss', 'Train Loss')
  load_plot('train_accuracy', 'Train Accuracy')
  load_plot('test_loss', 'Test Loss')
  load_plot('test_accuracy', 'Test Accuracy')