# src/safety/red_teaming/framework.py

import os
import json
import random
import datetime
from typing import List, Dict, Any, Optional, Callable

class RedTeamingFramework:
    """
    A framework for conducting red teaming exercises on language models.
    
    This framework helps organize, execute, and analyze various adversarial
    testing strategies designed to probe model boundaries and identify
    potential vulnerabilities.
    """
    
    def __init__(
        self,
        output_dir: str = "red_team_results",
        log_results: bool = True,
    ):
        """
        Initialize the red teaming framework.
        
        Args:
            output_dir: Directory to store red teaming results
            log_results: Whether to log results to disk
        """
        self.output_dir = output_dir
        self.log_results = log_results
        
        # Create output directory if it doesn't exist and logging is enabled
        if self.log_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Store different attack strategies
        self.attack_strategies = {}
        
        # Track red teaming results
        self.results = []
    
    def register_attack_strategy(
        self, 
        name: str, 
        strategy_fn: Callable[[str], List[str]]
    ) -> None:
        """
        Register an attack strategy function.
        
        Args:
            name: Name of the attack strategy
            strategy_fn: Function that generates adversarial inputs from a base prompt
        """
        self.attack_strategies[name] = strategy_fn
        
    def generate_adversarial_inputs(
        self, 
        base_prompts: List[str],
        strategy_name: Optional[str] = None,
        num_variations: int = 5
    ) -> Dict[str, List[str]]:
        """
        Generate adversarial inputs using registered strategies.
        
        Args:
            base_prompts: List of base prompts to modify
            strategy_name: Name of specific strategy to use (None = use all)
            num_variations: Number of variations to generate per base prompt
            
        Returns:
            Dictionary mapping strategy names to lists of generated inputs
        """
        adversarial_inputs = {}
        
        # Determine which strategies to use
        strategies = {}
        if strategy_name is not None:
            if strategy_name not in self.attack_strategies:
                raise ValueError(f"Unknown attack strategy: {strategy_name}")
            strategies[strategy_name] = self.attack_strategies[strategy_name]
        else:
            strategies = self.attack_strategies
        
        # Generate adversarial inputs for each strategy
        for name, strategy_fn in strategies.items():
            inputs_for_strategy = []
            
            for base_prompt in base_prompts:
                # Generate variations using the strategy
                variations = strategy_fn(base_prompt)
                
                # Limit to requested number of variations
                variations = variations[:num_variations]
                
                inputs_for_strategy.extend(variations)
            
            adversarial_inputs[name] = inputs_for_strategy
            
        return adversarial_inputs
    
    def evaluate_model_robustness(
        self,
        model_fn: Callable[[str], str],
        adversarial_inputs: Dict[str, List[str]],
        evaluation_fn: Callable[[str, str], Dict[str, Any]],
        model_name: str = "unnamed_model"
    ) -> Dict[str, Any]:
        """
        Evaluate model robustness against adversarial inputs.
        
        Args:
            model_fn: Function that takes a text input and returns model output
            adversarial_inputs: Dictionary mapping strategies to input lists
            evaluation_fn: Function that evaluates model output quality/safety
            model_name: Name identifier for the model
            
        Returns:
            Dictionary with evaluation results
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        evaluation_results = {
            "model_name": model_name,
            "timestamp": timestamp,
            "total_inputs": 0,
            "total_failures": 0,
            "strategies": {},
            "detailed_results": []
        }
        
        # Process each strategy
        for strategy_name, inputs in adversarial_inputs.items():
            strategy_results = {
                "total_inputs": len(inputs),
                "failures": 0,
                "failure_rate": 0.0,
            }
            
            # Process each input for this strategy
            for input_text in inputs:
                # Get model output
                try:
                    output = model_fn(input_text)
                except Exception as e:
                    output = f"ERROR: {str(e)}"
                
                # Evaluate the output
                eval_result = evaluation_fn(input_text, output)
                
                # Track failures
                if not eval_result.get("success", False):
                    strategy_results["failures"] += 1
                    evaluation_results["total_failures"] += 1
                
                # Store detailed result
                detailed_result = {
                    "strategy": strategy_name,
                    "input": input_text,
                    "output": output,
                    "evaluation": eval_result
                }
                evaluation_results["detailed_results"].append(detailed_result)
            
            # Calculate failure rate
            if strategy_results["total_inputs"] > 0:
                strategy_results["failure_rate"] = (
                    strategy_results["failures"] / strategy_results["total_inputs"]
                )
            
            # Add strategy results to overall results
            evaluation_results["strategies"][strategy_name] = strategy_results
            evaluation_results["total_inputs"] += strategy_results["total_inputs"]
        
        # Calculate overall failure rate
        if evaluation_results["total_inputs"] > 0:
            evaluation_results["failure_rate"] = (
                evaluation_results["total_failures"] / 
                evaluation_results["total_inputs"]
            )
        else:
            evaluation_results["failure_rate"] = 0.0
        
        # Log results if enabled
        if self.log_results:
            result_file = os.path.join(
                self.output_dir, 
                f"red_team_{model_name}_{timestamp}.json"
            )
            with open(result_file, "w") as f:
                json.dump(evaluation_results, f, indent=2)
        
        # Store results
        self.results.append(evaluation_results)
        
        return evaluation_results
    
    def generate_report(
        self, 
        results: Optional[Dict[str, Any]] = None,
        include_details: bool = False
    ) -> str:
        """
        Generate a human-readable report from evaluation results.
        
        Args:
            results: Results to report (uses latest results if None)
            include_details: Whether to include detailed input/output pairs
            
        Returns:
            Formatted report string
        """
        # Use latest results if none provided
        if results is None:
            if not self.results:
                return "No red teaming results available."
            results = self.results[-1]
        
        # Create the report
        report = f"# Red Team Evaluation Report\n\n"
        report += f"Model: {results['model_name']}\n"
        report += f"Timestamp: {results['timestamp']}\n\n"
        
        # Overall results
        report += "## Overall Results\n\n"
        report += f"- Total inputs tested: {results['total_inputs']}\n"
        report += f"- Total failures: {results['total_failures']}\n"
        report += f"- Overall failure rate: {results['failure_rate']:.2%}\n\n"
        
        # Results by strategy
        report += "## Results by Strategy\n\n"
        
        for strategy_name, strategy_results in results["strategies"].items():
            report += f"### {strategy_name}\n\n"
            report += f"- Inputs tested: {strategy_results['total_inputs']}\n"
            report += f"- Failures: {strategy_results['failures']}\n"
            report += f"- Failure rate: {strategy_results['failure_rate']:.2%}\n\n"
        
        # Include detailed results if requested
        if include_details and results["detailed_results"]:
            report += "## Detailed Results\n\n"
            
            for i, detail in enumerate(results["detailed_results"]):
                if not detail["evaluation"].get("success", True):
                    report += f"### Failure {i+1}: {detail['strategy']}\n\n"
                    report += f"**Input:**\n```\n{detail['input']}\n```\n\n"
                    report += f"**Output:**\n```\n{detail['output']}\n```\n\n"
                    
                    # Include evaluation details
                    report += "**Evaluation:**\n"
                    for key, value in detail["evaluation"].items():
                        if key != "success":
                            report += f"- {key}: {value}\n"
                    report += "\n"
        
        return report

def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Provides a framework for conducting red teaming exercises on language models",
        "key_classes": [
            {
                "name": "RedTeamingFramework",
                "purpose": "Framework for organizing, executing, and analyzing adversarial testing strategies",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, output_dir: str = \"red_team_results\", log_results: bool = True)",
                        "brief_description": "Initialize the red teaming framework with output configuration"
                    },
                    {
                        "name": "register_attack_strategy",
                        "signature": "register_attack_strategy(self, name: str, strategy_fn: Callable[[str], List[str]]) -> None",
                        "brief_description": "Register an attack strategy function for generating adversarial inputs"
                    },
                    {
                        "name": "generate_adversarial_inputs",
                        "signature": "generate_adversarial_inputs(self, base_prompts: List[str], strategy_name: Optional[str] = None, num_variations: int = 5) -> Dict[str, List[str]]",
                        "brief_description": "Generate adversarial inputs using registered strategies"
                    },
                    {
                        "name": "evaluate_model_robustness",
                        "signature": "evaluate_model_robustness(self, model_fn: Callable[[str], str], adversarial_inputs: Dict[str, List[str]], evaluation_fn: Callable[[str, str], Dict[str, Any]], model_name: str = \"unnamed_model\") -> Dict[str, Any]",
                        "brief_description": "Evaluate model robustness against adversarial inputs"
                    },
                    {
                        "name": "generate_report",
                        "signature": "generate_report(self, results: Optional[Dict[str, Any]] = None, include_details: bool = False) -> str",
                        "brief_description": "Generate a human-readable report from evaluation results"
                    }
                ],
                "inheritance": "",
                "dependencies": ["os", "json", "datetime", "typing"]
            }
        ],
        "external_dependencies": ["json", "datetime"],
        "complexity_score": 7  # High complexity due to the comprehensive framework
    }