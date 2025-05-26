#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_subpatcher_analyzer.py - Script to run the SubpatcherAnalyzer on Max/MSP patches

This script provides a convenient interface to run the SubpatcherAnalyzer on
one or more Max/MSP patch files, generating comprehensive analyses and visualizations
of their hierarchical structure.
"""

import os
import argparse
import logging
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from subpatcher_analyzer import SubpatcherAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"run_subpatcher_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_patch(file_path, output_dir, visualize=True, report=True, debug=False):
    """
    Process a single patch file with the SubpatcherAnalyzer
    
    Args:
        file_path: Path to the .maxpat file
        output_dir: Directory for output files
        visualize: Whether to generate visualizations
        report: Whether to generate reports
        debug: Whether to enable debug logging
        
    Returns:
        A dictionary with analysis results
    """
    try:
        # Create patch-specific output directory
        patch_id = os.path.basename(file_path).split('.')[0]
        patch_output_dir = os.path.join(output_dir, patch_id)
        os.makedirs(patch_output_dir, exist_ok=True)
        
        # Initialize analyzer
        analyzer = SubpatcherAnalyzer(debug=debug)
        
        # Load and analyze patch
        if analyzer.load_patch(file_path):
            # Analyze hierarchy
            logger.info(f"Analyzing hierarchy for {file_path}")
            analyzer.analyze_hierarchy(patch_id)
            
            # Analyze connections
            logger.info(f"Analyzing connections for {file_path}")
            analyzer.analyze_connections(patch_id)
            
            # Generate outputs
            results = {
                "patch_id": patch_id,
                "file_path": file_path,
                "output_dir": patch_output_dir,
                "hierarchy_stats": analyzer.hierarchy_stats,
                "connection_flows": analyzer.connection_flows,
                "output_files": []
            }
            
            if visualize:
                hierarchy_img = os.path.join(patch_output_dir, f"{patch_id}_hierarchy.png")
                flow_img = os.path.join(patch_output_dir, f"{patch_id}_flows.png")
                
                logger.info(f"Generating hierarchy visualization for {patch_id}")
                analyzer.visualize_hierarchy(patch_id, hierarchy_img)
                results["output_files"].append(hierarchy_img)
                
                logger.info(f"Generating connection flow visualization for {patch_id}")
                analyzer.visualize_connection_flows(patch_id, flow_img)
                results["output_files"].append(flow_img)
            
            if report:
                report_file = os.path.join(patch_output_dir, f"{patch_id}_report.json")
                graph_file = os.path.join(patch_output_dir, f"{patch_id}_graph.json")
                
                logger.info(f"Generating detailed report for {patch_id}")
                analyzer.generate_hierarchy_report(patch_id, report_file)
                results["output_files"].append(report_file)
                
                logger.info(f"Exporting hierarchy graph for {patch_id}")
                analyzer.export_hierarchy_to_networkx(patch_id, graph_file)
                results["output_files"].append(graph_file)
            
            logger.info(f"Analysis completed for {file_path}")
            return results
        else:
            logger.error(f"Failed to analyze {file_path}")
            return {
                "patch_id": patch_id,
                "file_path": file_path,
                "error": "Failed to load patch file"
            }
            
    except Exception as e:
        logger.exception(f"Error processing {file_path}: {str(e)}")
        return {
            "patch_id": os.path.basename(file_path).split('.')[0],
            "file_path": file_path,
            "error": str(e)
        }

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Run SubpatcherAnalyzer on Max/MSP patches")
    parser.add_argument("--input", "-i", required=True, 
                        help="Path to .maxpat file or directory containing .maxpat files")
    parser.add_argument("--pattern", "-p", default="**/*.maxpat", 
                        help="Glob pattern for finding .maxpat files (default: **/*.maxpat)")
    parser.add_argument("--output-dir", "-o", default="./subpatcher_analysis_output", 
                        help="Directory for output files")
    parser.add_argument("--no-visualize", "-nv", action="store_true", 
                        help="Disable visualization generation")
    parser.add_argument("--no-report", "-nr", action="store_true", 
                        help="Disable report generation")
    parser.add_argument("--debug", "-d", action="store_true", 
                        help="Enable debug logging")
    parser.add_argument("--parallel", "-j", type=int, default=1, 
                        help="Number of parallel processes to use (default: 1)")
    parser.add_argument("--summary-only", "-s", action="store_true",
                        help="Only generate summary report, skip individual analyses")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find patch files
    patch_files = []
    if os.path.isfile(args.input):
        if args.input.endswith('.maxpat'):
            patch_files = [args.input]
        else:
            logger.error(f"Input file {args.input} is not a .maxpat file")
            return
    elif os.path.isdir(args.input):
        pattern = os.path.join(args.input, args.pattern)
        patch_files = glob.glob(pattern, recursive=True)
    else:
        logger.error(f"Input {args.input} is not a file or directory")
        return
    
    if not patch_files:
        logger.error(f"No .maxpat files found at {args.input} with pattern {args.pattern}")
        return
    
    logger.info(f"Found {len(patch_files)} .maxpat files to analyze")
    
    # Process patch files
    results = []
    
    if args.parallel > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for file_path in patch_files:
                if args.summary_only:
                    # Skip individual analyses for summary-only mode
                    results.append({
                        "patch_id": os.path.basename(file_path).split('.')[0],
                        "file_path": file_path
                    })
                else:
                    # Submit for processing
                    future = executor.submit(
                        process_patch, 
                        file_path, 
                        args.output_dir, 
                        not args.no_visualize, 
                        not args.no_report,
                        args.debug
                    )
                    futures.append(future)
            
            # Collect results
            if not args.summary_only:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.exception(f"Error in worker process: {str(e)}")
    else:
        # Sequential processing
        for file_path in patch_files:
            if args.summary_only:
                # Skip individual analyses for summary-only mode
                results.append({
                    "patch_id": os.path.basename(file_path).split('.')[0],
                    "file_path": file_path
                })
            else:
                # Process sequentially
                result = process_patch(
                    file_path, 
                    args.output_dir, 
                    not args.no_visualize, 
                    not args.no_report,
                    args.debug
                )
                results.append(result)
    
    # Generate aggregate report if requested
    if len(results) > 1:
        # Create summary report
        import json
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Path for summary files
        summary_dir = os.path.join(args.output_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        logger.info("Generating summary report")
        
        # Save raw results
        with open(os.path.join(summary_dir, "analysis_results.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Prepare data for summary analysis
        valid_results = [r for r in results if 'error' not in r]
        
        if not args.summary_only and valid_results:
            # Extract hierarchy statistics
            hierarchy_data = []
            for result in valid_results:
                if 'hierarchy_stats' in result:
                    stats = result['hierarchy_stats']
                    hierarchy_data.append({
                        'patch_id': result['patch_id'],
                        'total_levels': stats.get('total_levels', 0),
                        'total_patchers': stats.get('total_patchers', 0),
                        'max_depth': stats.get('max_depth', 0),
                        'avg_boxes_per_patcher': stats.get('avg_boxes_per_patcher', 0)
                    })
            
            # Create DataFrame
            df_hierarchy = pd.DataFrame(hierarchy_data)
            
            # Save to CSV
            df_hierarchy.to_csv(os.path.join(summary_dir, "hierarchy_summary.csv"), index=False)
            
            # Generate hierarchy visualization
            if not args.no_visualize and len(df_hierarchy) > 0:
                plt.figure(figsize=(12, 8))
                
                # Patch depth distribution
                plt.subplot(2, 2, 1)
                depth_counts = df_hierarchy['max_depth'].value_counts().sort_index()
                depth_counts.plot(kind='bar')
                plt.title('Patch Depth Distribution')
                plt.xlabel('Hierarchy Depth')
                plt.ylabel('Number of Patches')
                
                # Patcher count distribution
                plt.subplot(2, 2, 2)
                bin_edges = np.linspace(0, df_hierarchy['total_patchers'].max() + 1, 10)
                plt.hist(df_hierarchy['total_patchers'], bins=bin_edges)
                plt.title('Subpatcher Count Distribution')
                plt.xlabel('Number of Subpatchers')
                plt.ylabel('Number of Patches')
                
                # Box count per patcher
                plt.subplot(2, 2, 3)
                bin_edges = np.linspace(0, df_hierarchy['avg_boxes_per_patcher'].max() + 1, 10)
                plt.hist(df_hierarchy['avg_boxes_per_patcher'], bins=bin_edges)
                plt.title('Average Boxes per Patcher')
                plt.xlabel('Avg Box Count')
                plt.ylabel('Number of Patches')
                
                # Correlation between depth and patcher count
                plt.subplot(2, 2, 4)
                plt.scatter(df_hierarchy['max_depth'], df_hierarchy['total_patchers'])
                plt.title('Depth vs. Patcher Count')
                plt.xlabel('Hierarchy Depth')
                plt.ylabel('Number of Subpatchers')
                
                plt.tight_layout()
                plt.savefig(os.path.join(summary_dir, "hierarchy_summary.png"), dpi=300, bbox_inches="tight")
                
                # Extract patcher type distribution
                patcher_types = {}
                for result in valid_results:
                    if 'hierarchy_stats' in result and 'patchers_by_type' in result['hierarchy_stats']:
                        types = result['hierarchy_stats']['patchers_by_type']
                        for ptype, count in types.items():
                            if ptype not in patcher_types:
                                patcher_types[ptype] = 0
                            patcher_types[ptype] += count
                
                if patcher_types:
                    plt.figure(figsize=(10, 6))
                    types = list(patcher_types.keys())
                    counts = list(patcher_types.values())
                    plt.bar(types, counts)
                    plt.title('Subpatcher Type Distribution')
                    plt.xlabel('Patcher Type')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(summary_dir, "patcher_types.png"), dpi=300, bbox_inches="tight")
        
        # List of errors
        errors = [r for r in results if 'error' in r]
        if errors:
            with open(os.path.join(summary_dir, "errors.json"), 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2)
            
            logger.warning(f"Encountered errors in {len(errors)} patch files")
        
        logger.info(f"Summary report generated in {summary_dir}")
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()