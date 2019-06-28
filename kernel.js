export function makeModel()
{
  /*
   * On définit un modèle simple séquentiel.
   * Le réseau essst constitué d'une succesion de deux couches (dont une cachée)
   */
  const model = tf.sequential();

  /*
   * Première couche du modèle (couche d'entrée)
   * - On décide de paramétrer la couche cachée avec 10 neurones
   * -> c'est le nombre ('units') d'unité de sortie de la couche
   * - On opte pour une fonction d'activation 'sigmoid'
   * - Le format d'entrée correspond au nombre de caracatéristiques
   */
  model.add(tf.layers.dense({
    units: 4,
    activation: 'sigmoid',
    inputShape: [4],
    useBias: true,
    kernelInitializer: 'zeros',
    biasInitializer: 'zeros'
  }));

  /*
   * Deuxième couche du modèle (couche cachée)
   * - Le nombre d'unités de sortie est 1 (on attend une valeur : 0, 1 ou 2)
   * Pas de fonction d'activation --> on cherche la valeur exacte
   */
  model.add(tf.layers.dense({
    units: 1,
  }));

  /*
   * Affihage de la structure du modèle dans la console
   */
  model.summary();

  /*
   * Compilation du modèle
   *
   */
  const optimizer = tf.train.adam(0.05);
  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    // metrics: ['accuracy'],
  });

  return model;
}

/**
 * Train a `tf.Model` to recognize Iris flower type.
 *
 * @param model
 * @param xTrain Training feature data, a `tf.Tensor` of shape
 *   [numTrainExamples, 4]. The second dimension include the features
 *   petal length, petalwidth, sepal length and sepal width.
 * @param yTrain One-hot training labels, a `tf.Tensor` of shape
 *   [numTrainExamples, 3].
 * @param xTest Test feature data, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest One-hot test labels, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 * @returns The trained `tf.Model` instance.
 */
export async function trainModel(model, xTrain, yTrain, xTest, yTest) {

  // Call `model.fit` to train the model.
  const history = await model.fit(xTrain, yTrain, {
    epochs: 10,
    // validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, log) => { console.log(epoch); }
    }
  });

  displayWeights(model);
  validate(model, xTrain, yTrain);

}

/*
 * Affichage des valeurs des poids et des biais
 */
export const displayWeights = (model) => {
  model.weights.forEach(w => {
    console.log(w.name, w.shape, w.val.dataSync());
  });
}

/*
 * Validation du modèle sur l'ensemble de test
 */
function validate(model, xTrain, yTrain)
{
  tf.tidy(() => {
    let predictions = model.predict(xTrain).dataSync();
    let given = yTrain.dataSync();
    let error = 0;
    for (let i = 0; i <  predictions.length; i++) {
      error += Math.pow((predictions[i] - given[i]), 2);
      console.log(predictions[i] - given[i]);

      console.log("prediction " + predictions[i]);
    }
    console.log(`Erreur moyenne : ${Math.sqrt(error/predictions.length)}`);

    let success = 0;
    for( let j = 0; j < predictions.length; j++) {
      if(Math.round(predictions[j]) == given[j]) {
        success += 1;
      }
    }
    console.log("success: " + ((success/predictions.length)*100) + " %");
  })
}
