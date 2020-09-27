
import json
import os
import numpy as np

inputPath = '../input'
def saveExperiment(foldername, experiment):
    path = inputPath + '/' + foldername
    if not os.path.exists(path):
        os.mkdir(path)
    nu = experiment['nu']
    file = path + "/" + str(nu)+'Hz.json'
    data = json.dumps(experiment)
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

def loadExperiment(foldername, nu):
    file = inputPath + '/' + foldername + '/' + str(nu) + 'Hz.json'
    with open(file) as json_file:
        data = json.load(json_file)
    data = json.loads(data)
    return data

def buildEobs(grid, experiment, angle, Pj):
    # Note input grid is different to grid which is in attribute experiment. Grid in experiment was the discretisation for the simulation grid
    if grid['discretisationMethod'] == 'FinDiff':
        x, y, delx  = grid['x'], grid['y'], grid['delx'] # grid dimensions in metres
        M, N, L = grid['M'], grid['N'], grid['L'] # Grid dimensions in cell numbers

        Eobs = np.zeros(L, dtype = complex)
        EobsReal = np.array(experiment['measures'][str(float(angle))]['realPart'])
        EobsImag = np.array(experiment['measures'][str(float(angle))]['imagPart'])
        Eobs[Pj == 1] = EobsReal +  EobsImag*complex(0, 1)
        return Eobs
    if grid['discretisationMethod'] == 'FinEl':
        pass

# def SaveObservations(observations):
#     pass
#
# def SaveAEI():
#     pass
