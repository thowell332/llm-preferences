#!/usr/bin/env python3

import asyncio
import json
import os
import re
import time
import argparse
import sys
sys.path.append("../../")
from compute_utilities.compute_utilities import compute_utilities
from compute_utilities import templates as prompt_templates

async def optimize_utility_model(args):
    """
    Compute utilities for all options in a given options file and save them in a structured directory.
    """
    start_time = time.time()

    # Load options
    with open(args.options_path, 'r') as f:
        options_data = json.load(f)

    roles = [r.strip() for r in args.roles.split(",") if r.strip()] if args.roles else []

    template = None
    if args.comparison_prompt_template_key:
        if not hasattr(prompt_templates, args.comparison_prompt_template_key):
            raise ValueError(
                f"Unknown template key: {args.comparison_prompt_template_key}. "
                f"Expected a symbol in compute_utilities/templates.py."
            )
        template = getattr(prompt_templates, args.comparison_prompt_template_key)

    if roles:
        if template is None:
            template = (
                prompt_templates.comparison_prompt_template_reasoning_role_default
                if args.with_reasoning
                else prompt_templates.comparison_prompt_template_role_default
            )

        print(f"\nComputing utilities for {args.options_path} across {len(roles)} role(s)...")
        utility_results = {"roles": roles, "results_by_role": {}}
        for role_name in roles:
            print(f"\nRole: {role_name}")
            resolved_template = template.replace("{role}", role_name)
            role_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", role_name).strip("_").lower() or "role"
            role_save_suffix = f"{args.save_suffix}_{role_slug}" if args.save_suffix else role_slug

            utility_results["results_by_role"][role_name] = await compute_utilities(
                options_list=options_data,
                model_key=args.model_key,
                create_agent_config_path=args.create_agent_config_path,
                create_agent_config_key=args.create_agent_config_key,
                compute_utilities_config_path=args.compute_utilities_config_path,
                compute_utilities_config_key=args.compute_utilities_config_key,
                comparison_prompt_template=resolved_template,
                save_dir=args.save_dir,
                save_suffix=role_save_suffix,
                with_reasoning=args.with_reasoning
            )
    else:
        print(f"\nComputing utilities for {args.options_path}...")
        utility_results = await compute_utilities(
            options_list=options_data,
            model_key=args.model_key,
            create_agent_config_path=args.create_agent_config_path,
            create_agent_config_key=args.create_agent_config_key,
            compute_utilities_config_path=args.compute_utilities_config_path,
            compute_utilities_config_key=args.compute_utilities_config_key,
            comparison_prompt_template=template,
            save_dir=args.save_dir,
            save_suffix=args.save_suffix,
            with_reasoning=args.with_reasoning
        )

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    return utility_results

async def main():
    parser = argparse.ArgumentParser(description="Compute and save utilities for a given options file.")
    parser.add_argument("--model_key", default="gpt-4o", help="Model key to use")
    parser.add_argument("--save_dir", default="../shared_utilities", help="Base directory to save results")
    parser.add_argument("--save_suffix", default=None, help="Custom suffix for saved files")
    parser.add_argument("--options_path", default="../shared_options/options_hierarchical_v1.json", help="Path to options file")
    parser.add_argument("--with_reasoning", action="store_true", help="Whether to use reasoning in prompts")
    parser.add_argument("--compute_utilities_config_path", default="../compute_utilities.yaml", help="Path to compute_utilities.yaml")
    parser.add_argument("--compute_utilities_config_key", default="default", help="Key to use in compute_utilities.yaml")
    parser.add_argument("--create_agent_config_path", default="../create_agent.yaml", help="Path to create_agent.yaml")
    parser.add_argument("--create_agent_config_key", default=None, help="Key to use in create_agent.yaml (if None, uses 'default_with_reasoning' if with_reasoning=True, else 'default')")
    parser.add_argument("--roles", default=None, help="Comma-separated roles (e.g., 'judge,doctor'). If set, runs one utility computation per role.")
    parser.add_argument(
        "--comparison_prompt_template_key",
        default=None,
        help="Optional template variable name from compute_utilities/templates.py."
    )
    args = parser.parse_args()

    await optimize_utility_model(args)

if __name__ == "__main__":
    asyncio.run(main()) 