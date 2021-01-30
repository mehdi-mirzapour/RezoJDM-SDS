# RezoJDM-SDS (Semantic DataSet)
This is a subproject related to the creation of RezoJDM-SDS, French Semantic Relation DataSet with 10 semantic types. RezoJDM-SDS is extracted under several constraints from RezoJDM a French lexical-semantic network.

## Instruction:  

Simply download the `datasets` folder and use it for training machine learning models.


## RezoJDM-SDS: A Semantic DataSet built from RezoJDM

RezoJDM-SDS is a Semantic DataSet created from RezoJDM. Among wide variety of possibilities, we have only focused on 10 most frequent semantic relations which belong to the ontological and predicative categories of RezoJDM and are more common in semantic linguistic analysis. We have selected reliable relations with weights greater than 50 in RezoJDM. This constraint guarantees to select positive relations that have been validated more frequently by different players in the JDM game. Relations with weights less than 50 are treated as negative in our dataset. The descriptions and examples of the relation types are illustrated in tables. RezoJDM-SDS is gained by randomly splitting the initial dataset into two train and test samples (80\% and 20\%). 

<p align="center">
  <img src="https://github.com/mehdi-mirzapour/RezoJDM-SDS/blob/main/resources/Table_1.jpg" width="500" height="350">
  <br>
  <img src="https://github.com/mehdi-mirzapour/RezoJDM-SDS/blob/main/resources/Table_2.jpg" width="500" height="350">
  <img src="https://github.com/mehdi-mirzapour/RezoJDM-SDS/blob/main/resources/Table_3.jpg" width="500" height="200">
</p>


## Citations
```bibtex
@inproceedings{lafourcade2007making,
  title={Making people play for Lexical Acquisition with the JeuxDeMots prototype},
  author={Lafourcade, Mathieu},
  booktitle={SNLP'07: 7th international symposium on natural language processing},
  pages={7},
  year={2007}
}
```

```bibtex
@inproceedings{mirzapour2021,
  TITLE = {Improving Quality of French Indirect Coreference Resolver by Employing Semantic Features from RezoJDM},
  AUTHOR = {Mirzapour, Mehdi and Ragheb, Waleed and Cousot, Kevin and Lafourcade, Mathieu and Jacquenet, Hélène and Carbon, Lawrence},
  BOOKTITLE = {Under Review}
}
```

### License
RezoJDM-SDS is MIT-licensed, as can be found in the LICENSE file.

