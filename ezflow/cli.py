import argparse
import sys
from pathlib import Path

def create_project(name: str):
    """Create a new EZFlow project with standard directory structure."""
    project_dir = Path(name)
    
    # Create project directory
    project_dir.mkdir(exist_ok=True)
    
    # Create standard directories
    (project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (project_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "experiments").mkdir(exist_ok=True)
    (project_dir / "notebooks").mkdir(exist_ok=True)
    
    # Create basic files
    (project_dir / ".gitignore").write_text("""
# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/

# Virtual Environment
venv/
env/

# IDE
.idea/
.vscode/

# ML specific
models/
experiments/
mlruns/

# Data
data/
*.csv
*.json
*.jsonl

# OS specific
.DS_Store
Thumbs.db
""")
    
    print(f"Created new EZFlow project in {project_dir.absolute()}")

def main():
    """Main entry point for the EZFlow CLI."""
    parser = argparse.ArgumentParser(description="EZFlow CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Create project command
    create_parser = subparsers.add_parser("create", help="Create a new EZFlow project")
    create_parser.add_argument("name", help="Name of the project")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "create":
        create_project(args.name)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 