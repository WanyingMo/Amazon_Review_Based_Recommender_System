from multiprocessing import Pool, cpu_count, Pipe, Process
from tqdm import tqdm
import numpy as np
import json
import csv
import random

def splitWorkerInitializer(l):
    global selectedKeys
    selectedKeys = l

def splitWorker(line):
    row = json.loads(line)
    return [row.get(key) for key in selectedKeys]

def splitDataset(dataPath:str, totalSize:int, selectedKeys:list, outputPath:str, processNum:int = cpu_count(), chunkSize:int = 1024, printParameters:bool = False):
    """
    multiprocess split function

    Args:
        dataPath (str): original dataset path
        totalSize (int): total size of original dataset
        selectedKeys (list): a list containing selected keys
        outputPath (str): output dataset path
        processNum (int, optional): number of sub-processes. Defaults to cpu_count().
        chunkSize (int, optional): the size of each chunk splitted from the iterable. Defaults to 1024.
        printParameters (bool, optional): show the parameters or not. Defaults to False.
    """

    if printParameters:
        print(f"Dataset File: {dataPath}")
        print(f"Total Size: {totalSize}")
        print(f"Selected Keys: {selectedKeys}")
        print(f"Output File: {outputPath}")
        print(f"Number of processes: {processNum}")
        print(f"Size of each chunk: {chunkSize}")
    
    pool = Pool(processNum, splitWorkerInitializer, (selectedKeys,))
    with open(dataPath, 'r', encoding="utf-8") as file:
        with open(outputPath, 'w', newline='') as outputFile:
            csvWriter = csv.writer(outputFile, selectedKeys)
            csvWriter.writerow(selectedKeys) # write header
            with tqdm(total=totalSize, desc="Processing", unit_scale=True) as pbar:
                for row in pool.imap(splitWorker, file, chunkSize):
                    csvWriter.writerow(row)
                    pbar.update()
    pool.close()
    pool.join()


def consumer(pipe, outputPath, sampleSize):
    _outPipe, _inPipe = pipe
    _inPipe.close()

    # with tqdm(total=sampleSize, desc="Processing", leave=True, unit_scale=True) as pbar:
    with open(outputPath, 'w', newline='') as outputFile:
        csvWriter = csv.writer(outputFile)
        while True:
            try:
                line = _outPipe.recv()
                csvWriter.writerow(line)
                # pbar.update()
            except EOFError:
                break

def sampleDataset(dataPath:str, totalSize:int, outputPath:str, sampleSize:int, printParameters:bool = False):
    """
    multiprocess sample function

    Args:
        dataPath (str): original dataset path
        totalSize (int): total size of original dataset
        outputPath (str): output dataset path
        sampleSize (int): size of sample dataset
        printParameters (bool, optional): show the parameters or not. Defaults to False.
    """

    if printParameters:
        print(f"Dataset File: {dataPath}")
        print(f"Total Size: {totalSize}")
        print(f"Output File: {outputPath}")
        print(f"Sample size: {sampleSize}")

    outPipe, inPipe = Pipe()

    consumerPc = Process(target = consumer, args = ((outPipe, inPipe), outputPath, sampleSize))
    consumerPc.start()

    sampleIndicesList = sorted(random.sample(range(totalSize), sampleSize))

    outPipe.close()
    with tqdm(total=sampleSize, desc="Processing", leave=True, unit_scale=True) as pbar:
        with open(dataPath, encoding="utf-8", newline='') as file:
            csvReader = csv.reader(file)
            csvHeader = next(csvReader)
            inPipe.send(csvHeader)

            sampleIndex = 0
            for i, line in enumerate(csvReader):
                if i == sampleIndicesList[sampleIndex]:
                    inPipe.send(line)
                    sampleIndex += 1
                    pbar.update()
                if sampleIndex >= sampleSize:
                    break
    inPipe.close()
    consumerPc.join()

if __name__ == '__main__':
    input_csv_path = "Data/All_Amazon_Review_User_Item_Rating.csv"
    output_csv_path = "Data/test.csv"
    total_records = 157260921
    required_records = 2500000
    processNum = cpu_count()
    chunkSize = 1024
    try:
        sampleDataset(input_csv_path, total_records, output_csv_path, required_records, True)
    except Exception as e:
        print(e)
        exit()

    # selectedKeys = ['reviewerID', 'asin', 'overall']
    # dataPath = "Data/All_Amazon_Review_5.json"
    # totalSize = 157260921
    # outputPath = "Data/test.csv"
    # processNum = cpu_count() # number of processes (customize this variable based on different CPUs)
    # chunkSize = 1024 # the size of each chunk splitted from the iterable
    # try:
    #     splitDataset(dataPath, totalSize, selectedKeys, outputPath, processNum, chunkSize, True)
    # except Exception as e:
    #     print(e)
    #     exit()
