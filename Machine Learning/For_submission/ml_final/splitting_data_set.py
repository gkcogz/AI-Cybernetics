import os
import random
import shutil

def split_dataset(data_dir, train_dir, test_dir, split_ratio=0.2):
    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load the truth file
    with open(os.path.join(data_dir, 'truth.dsv'), 'r') as f:
        lines = f.readlines()
    
    # Shuffle the lines
    random.shuffle(lines)
    
    # Split the lines
    split_index = int(len(lines) * (1 - split_ratio))
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]
    
    # Write the truth files for train and test sets
    with open(os.path.join(train_dir, 'truth.dsv'), 'w') as f:
        f.writelines(train_lines)
    
    with open(os.path.join(test_dir, 'truth.dsv'), 'w') as f:
        f.writelines(test_lines)
    
    # Copy images to the train and test directories
    for line in train_lines:
        fname = line.split(':')[0]
        shutil.copy(os.path.join(data_dir, fname), os.path.join(train_dir, fname))
    
    for line in test_lines:
        fname = line.split(':')[0]
        shutil.copy(os.path.join(data_dir, fname), os.path.join(test_dir, fname))

# Example usage
data_dir = "C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\train_1000_10\\train_1000_10"
train_dir = "C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\train_1000_10\\train"
test_dir = "C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\train_1000_10\\test"
split_dataset(data_dir, train_dir, test_dir, split_ratio=0.2)
