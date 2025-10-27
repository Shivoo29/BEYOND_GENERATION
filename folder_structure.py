import os

def get_tree(path, prefix=""):
    """Recursively generates the folder structure as text."""
    tree_str = ""
    items = sorted(os.listdir(path))
    
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        connector = "├── " if i < len(items) - 1 else "└── "
        tree_str += prefix + connector + item + "\n"
        
        if os.path.isdir(item_path):
            extension = "│   " if i < len(items) - 1 else "    "
            tree_str += get_tree(item_path, prefix + extension)
    
    return tree_str

if __name__ == "__main__":
    folder_path = input("Enter folder path (leave blank for current folder): ").strip()
    if not folder_path:
        folder_path = os.getcwd()

    print(f"\n📁 Folder Structure of: {folder_path}\n")

    structure = get_tree(folder_path)
    print(structure)

    # Save to file
    output_file = os.path.join(folder_path, "structure.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Folder Structure of: {folder_path}\n\n")
        f.write(structure)

    print(f"✅ Structure saved to: {output_file}")

