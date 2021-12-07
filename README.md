Model should assign probabilities to DRs already explicitly introduced in preceding text fragment, but also reserve some
probability mass for 'new' DRs.

In incremental model, predicate is not always available in the history h^(t) for subject NPs.

EntityNLM
- RNN, augmented with random variables for entity mentions that capture coreference
- Dynamic representations of entities
- Generative

Additional random variables and representations for entities:

<img src="https://render.githubusercontent.com/render/math?math=R_t"> 

- Binary random variable, indicates whether <img src="https://render.githubusercontent.com/render/math?math=x_t"> belongs to an entity mention.

<img src="https://render.githubusercontent.com/render/math?math=L_t \in \{1, ....,\ell_{max}\}"> 

- Categorical random variable if <img src="https://render.githubusercontent.com/render/math?math=R_t = 1">
- Indicates number of remaining words in the mention, including current word
- <img src="https://render.githubusercontent.com/render/math?math=L_t = 1"> for the last word in a mention

<img src="https://render.githubusercontent.com/render/math?math=E_t \in \varepsilon_t">

- Index of the entity referred to, if <img src="https://render.githubusercontent.com/render/math?math=R_t = 1">
- <img src="https://render.githubusercontent.com/render/math?math=\varepsilon_t"> starts as <img src="https://render.githubusercontent.com/render/math?math={1}">, grows monotonically with <img src="https://render.githubusercontent.com/render/math?math=t">


<img src="https://render.githubusercontent.com/render/math?math=e_{i,t}">

- Vector of entity <img src="https://render.githubusercontent.com/render/math?math=i"> at timestep <img src="https://render.githubusercontent.com/render/math?math=t">

![Random variable examples](img/random_vars.png?raw=true)

## Dynamic Entity Representations 
Before predicting the entity at timestep t, we need a new embedding for the entity, if it does not exist.

![Dynamic entity representations](img/dynamic_ent_repr.png?raw=true)
`initialize_entity_embedding()`