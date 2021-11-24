import json
import matplotlib.pyplot as plt


def load_plot(propertyName, graphTitle):
  fig, ax = plt.subplots()
  for i in range(3, 10):
    f = open(f'./json/Modelo{i}.json')

    data = json.load(f)

    ax.plot(data['epochs'], data[propertyName], label=f'Modelo-{i}')

  ax.legend(loc=2, bbox_to_anchor=(0.80, 1.0))
  plt.title(graphTitle)

  plt.show()
  

if __name__=='__main__':
  
  load_plot('train_loss', 'Train Loss')
  load_plot('train_accuracy', 'Train Accuracy')
  load_plot('test_loss', 'Test Loss')
  load_plot('test_accuracy', 'Test Accuracy')