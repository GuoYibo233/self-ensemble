"""Interactive parameter input helpers."""

import sys
from typing import Optional, List


def prompt_for_parameter(param_name: str, description: str, default: Optional[str] = None) -> str:
    """
    Prompt user for a parameter value.
    
    Args:
        param_name: Name of the parameter
        description: Description of what this parameter does
        default: Default value if user presses Enter
        
    Returns:
        User-provided value or default
    """
    if default:
        prompt = f"{description} [{default}]: "
    else:
        prompt = f"{description}: "
    
    value = input(prompt).strip()
    if not value and default:
        return default
    elif not value:
        print(f"Error: {param_name} is required.")
        sys.exit(1)
    return value


def select_from_options(param_name: str, description: str, options: List[str], 
                       default: Optional[str] = None) -> str:
    """
    Prompt user to select from a list of options.
    
    Args:
        param_name: Name of the parameter
        description: Description of what this parameter does
        options: List of valid options
        default: Default option if user presses Enter
        
    Returns:
        Selected option
    """
    print(f"\n{description}:")
    for i, option in enumerate(options, 1):
        marker = " (default)" if option == default else ""
        print(f"  {i}. {option}{marker}")
    
    if default:
        prompt = f"Select option [1-{len(options)}] (default: {default}): "
    else:
        prompt = f"Select option [1-{len(options)}]: "
    
    while True:
        choice = input(prompt).strip()
        
        if not choice and default:
            return default
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            else:
                print(f"Error: Please select a number between 1 and {len(options)}.")
        except ValueError:
            print("Error: Please enter a valid number.")


def confirm_action(message: str, default: bool = True) -> bool:
    """
    Ask user for yes/no confirmation.
    
    Args:
        message: Question to ask
        default: Default answer if user presses Enter
        
    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    prompt = f"{message} [{default_str}]: "
    
    while True:
        choice = input(prompt).strip().lower()
        
        if not choice:
            return default
        elif choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Error: Please answer 'y' or 'n'.")
