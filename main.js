import * as Tensors from './prepare.js';
import * as Kernel from './kernel.js';

/*
 * Initialisation de variables
 */
let xTrain, yTrain, xTest, yTest;

/*
* Programme principal
*/

/*
 * Création des tenseurs à partir des données du fichier iris.js
 * Cela crée quatre tenseurs :
 * - xTrain : Les caractéristiques pour l'entrainement
 * - xTest : Les caractéristiques pour la validdation
 * - yTrain : Les étiquettes pour l'entrainement
 * - yTest : Les étiquettes pour la validdation
 */
[xTrain, yTrain, xTest, yTest] = Tensors.getIrisData(0.2);

/*
 * Création du modèle
 */
let model = Kernel.makeModel();

/*
 * Entraînement du modèle
 * La validation, à partir du jeu de test est enchaînée à l'entraînement
 */
Kernel.trainModel(model, xTrain, yTrain, xTest, yTest);
