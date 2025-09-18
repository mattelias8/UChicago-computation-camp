import scriptflow as sf
import json

# set main options
sf.init({
    # "executors":{
    #     "hpc": {
    #         "maxsize" : 5
    #     } 
    # },
    "executors":{
        "local": {
            "maxsize" : 5
        } 
    },
    'debug':True
})

# example of a simple step that combines outcomes
def step2_combine_file():
    with open('test_1.txt') as f:
        a = int(f.readlines()[0])
    with open('test_2.txt') as f:
        b = int(f.readlines()[0])
    with open('final.txt','w') as f:
        f.write("{}\n".format(a+b))

# define a flow called sleepit
async def flow_mcbs():

    # prefix = "conda run -n camp "  # if you want to use conda environment
    prefix = ""

    tasks = [
        sf.Task(
            cmd    = [f"{prefix}python scripts/bootstrap.py -o output/bootstrap/bs_{i}.json -i {i}"],
            shell = True,
            outputs = f"output/bootstrap/bs_{i}.json",
            name   = f"solve-{i}")
        for i in range(5)
    ]

    # we make sure all the tasks are executed
    await sf.bag(*tasks)

    # we combine all outputs into one json
    combined = []
    for t in tasks:
        with open(t.get_outputs()[0]) as f:
            combined.extend(json.load(f))

    with open("output/bootstrap/bs.json", "w") as f:
        json.dump(combined, f, indent=2)
