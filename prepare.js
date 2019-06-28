import * as Iris from './iris.js';

/**
 * Prépare les tenseurs à partir de tableaux de données
 * La fonction mélange aléatoirement les données pour que l'apprentissage soit homogène
 * Elle sépare, en fonctino de la valeur de 'testSplit', deux jeux d'exemples :
 * - un pour l'entraînement
 * - un pour la validation du modèle
 *
 *
 */
function makeTensors(features, labels, testSplit) {
  /*
   * Si les paramètres et les labels ne son pas en mêmes nombres, c'est une erreur
   */
  const numExamples = features.length;
  if (numExamples !== labels.length) {
    throw new Error('data and split have different numbers of examples');
  }

  /*
   * Fonction pour ordonner aléatoirement les xemples d'apprentissage
   */
  const indices = [];
  for (let i = 0; i < numExamples; ++i) {
    indices.push(i);
  }
  // Ordonnancement aléatoire des indice de tableau
  tf.util.shuffle(indices);
  // Production de tabelaux des caractéristiques et des labels ordonnancés aléatoirement
  const shuffledFeatures = [];
  const shuffledLabels = [];
  for (let i = 0; i < numExamples; ++i) {
    shuffledFeatures.push(features[indices[i]]);
    shuffledLabels.push(labels[indices[i]]);
  }

  console.log(('Tableau des exemples mélangés : '));
  console.log(shuffledFeatures);

  /*
   * Calcul du nombre d'exemples pour l'apprentissage et la validation
   * en fonction du pourcentage décidé
   */
  const numTestExamples = Math.round(numExamples * testSplit);
  const numTrainExamples = numExamples - numTestExamples;

  /*
   * Calcul du nombre de caractéristiques par exemple
   */
  const xDims = shuffledFeatures[0].length;

  const xs = tf.tensor(shuffledFeatures);
  // xs.print();

  /* Variante de la création du tenseur en utilisant la méthode 'tensor2D'
  * On soit alors fournir les dimensions de la matrice
  *
  * const xs = tf.tensor2d(shuffledFeatures, [numExamples, xDims]);
  */

  const ys = tf.tensor(shuffledLabels);
  // ys.print();

  // Séparation du jeu de données en deux sous-ensembles
  const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
  xTrain.print();
  const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
  xTest.print();
  const yTrain = ys.slice(0, numTrainExamples);
  yTrain.print();
  const yTest = ys.slice(numTestExamples);
  yTest.print();
  return [xTrain, yTrain, xTest, yTest];
}

/**
 * Séparation des caractéristiques et des étiquettes.
 * Les exemples sont donnés sous la forme : [7.5, 2.4, 3.1, 1.8, 1]
 * où les quatre premières valeur sont les caractéristiques et la dernière l'étiquette
 */
export function getIrisData(testSplit) {
  return tf.tidy(() => {
    /*
     * Séparation des caractéristiques et des étiquettes
     * et création de sous-ensesmbles
     * 1) Initialisation de tableaux
     * 2) Répartition des exemples et des étiquettes dans les différents tableaux
     */

    const features = [];
    const labels = [];

    /*
     * Tenseurs pour les différentes dimensions
     */

    for (const example of Iris.IRIS_DATA) {
      console.log(example);
      const label = example[example.length - 1];
      const sample = example.slice(0, example.length - 1);
      features.push(sample);
      labels.push(label);
    }
      console.log(features, labels);
    /*
     * Conversion des différents tableaux en tenseurs
     */
    const [xTrain, yTrain, xTest, yTest] = makeTensors(features, labels, testSplit);
    // console.log(xTrain.argMax(-1).dataSync());
    return [xTrain, yTrain, xTest, yTest];
  });
}
