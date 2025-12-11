#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools CLI - Command Line Interface

Usage:
    hamtool freq --mhz 7.1 [--ssz 0.01]
    hamtool lambda --m 40
    hamtool antenna dipole --mhz 7.1
    hamtool antenna vertical --mhz 14.2
    hamtool antenna yagi --mhz 14.2 --elements 5 --boom 6.5
    hamtool feedline loss --mhz 14.2 --cable RG-58 --length 30
    hamtool feedline compare --mhz 14.2 --length 30
    hamtool erp --p-tx 100 --gain-dbd 3 --loss-db 2
    hamtool db --ratio 2
    hamtool muf --fof2 5.0 --distance 3000 --height 300
    hamtool field --p-tx 100 --gain-dbi 6 --km 1 --mhz 14.2
    hamtool ssz --mhz 7.1 --delta 0.01

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import argparse
import sys
from typing import Optional

from . import core
from . import antennas
from . import feedline
from . import propagation
from . import field_strength
from . import ssz_extension


def cmd_freq(args):
    """Handle frequency/wavelength command."""
    if args.mhz:
        f_mhz = args.mhz
    elif args.khz:
        f_mhz = args.khz / 1000
    elif args.hz:
        f_mhz = args.hz / 1e6
    else:
        print("Error: Specify frequency with --mhz, --khz, or --hz")
        return 1
    
    result = core.calculate_frequency_info(f_mhz)
    
    print(f"\n{'='*50}")
    print(f"Frequency / Wavelength Calculation")
    print(f"{'='*50}")
    print(result)
    
    # SSZ mode
    if args.ssz is not None:
        delta_seg = args.ssz
        ssz_result = ssz_extension.compare_lambda_with_ssz(f_mhz, delta_seg)
        print(f"\n{'='*50}")
        print(f"SSZ Expert Mode (delta_seg = {delta_seg:.4f})")
        print(f"{'='*50}")
        print(f"Classical:    {ssz_result.classical_value:.4f} m")
        print(f"SSZ:          {ssz_result.ssz_value:.4f} m")
        print(f"Difference:   {ssz_result.difference_percent:+.4f}%")
        print(f"\nInterpretation: SSZ segmentation slightly reduces")
        print(f"the effective wave speed, shortening wavelength.")
    
    return 0


def cmd_lambda(args):
    """Handle wavelength to frequency command."""
    if args.m:
        lambda_m = args.m
    else:
        print("Error: Specify wavelength with --m")
        return 1
    
    f_mhz = core.lambda_to_freq_mhz(lambda_m)
    result = core.calculate_frequency_info(f_mhz)
    
    print(f"\n{'='*50}")
    print(f"Wavelength to Frequency")
    print(f"{'='*50}")
    print(result)
    
    return 0


def cmd_antenna(args):
    """Handle antenna calculations."""
    if args.antenna_type == "dipole":
        if not args.mhz:
            print("Error: Specify frequency with --mhz")
            return 1
        
        k = args.k if args.k else 0.95
        result = antennas.calculate_dipole(args.mhz, k)
        print(f"\n{result}")
        
        # SSZ mode
        if args.ssz is not None:
            ssz_result = ssz_extension.compare_antenna_length_with_ssz(
                args.mhz, args.ssz, "dipole", k
            )
            print(f"\n{'='*50}")
            print(f"SSZ Expert Mode (delta_seg = {args.ssz:.4f})")
            print(f"{'='*50}")
            print(f"Classical length: {ssz_result.classical_value:.4f} m")
            print(f"SSZ length:       {ssz_result.ssz_value:.4f} m")
            print(f"Difference:       {ssz_result.difference_percent:+.4f}%")
    
    elif args.antenna_type == "vertical":
        if not args.mhz:
            print("Error: Specify frequency with --mhz")
            return 1
        
        k = args.k if args.k else 0.95
        result = antennas.calculate_vertical(args.mhz, k)
        print(f"\n{result}")
        
        # SSZ mode
        if args.ssz is not None:
            ssz_result = ssz_extension.compare_antenna_length_with_ssz(
                args.mhz, args.ssz, "vertical", k
            )
            print(f"\n{'='*50}")
            print(f"SSZ Expert Mode (delta_seg = {args.ssz:.4f})")
            print(f"{'='*50}")
            print(f"Classical length: {ssz_result.classical_value:.4f} m")
            print(f"SSZ length:       {ssz_result.ssz_value:.4f} m")
            print(f"Difference:       {ssz_result.difference_percent:+.4f}%")
    
    elif args.antenna_type == "yagi":
        if not args.mhz or not args.elements or not args.boom:
            print("Error: Specify --mhz, --elements, and --boom")
            return 1
        
        result = antennas.calculate_yagi(args.mhz, args.elements, args.boom)
        print(f"\n{result}")
    
    else:
        print(f"Unknown antenna type: {args.antenna_type}")
        return 1
    
    return 0


def cmd_feedline(args):
    """Handle feedline calculations."""
    if args.feedline_cmd == "loss":
        if not args.mhz or not args.cable or not args.length:
            print("Error: Specify --mhz, --cable, and --length")
            return 1
        
        p_tx = args.p_in_watt if args.p_in_watt else None
        result = feedline.calculate_feedline_loss(
            args.mhz, args.cable, args.length, p_tx
        )
        print(f"\n{result}")
    
    elif args.feedline_cmd == "compare":
        if not args.mhz or not args.length:
            print("Error: Specify --mhz and --length")
            return 1
        
        p_tx = args.p_in_watt if args.p_in_watt else 100.0
        result = feedline.compare_cables(args.mhz, args.length, p_tx)
        print(f"\n{result}")
    
    elif args.feedline_cmd == "cables":
        print("\nAvailable cable types:")
        for cable in feedline.get_available_cables():
            info = feedline.CABLE_INFO.get(cable, {})
            desc = info.get("description", "")
            print(f"  {cable:<15} - {desc}")
    
    else:
        print(f"Unknown feedline command: {args.feedline_cmd}")
        return 1
    
    return 0


def cmd_erp(args):
    """Handle ERP/EIRP calculation."""
    if not args.p_tx:
        print("Error: Specify --p-tx")
        return 1
    
    gain = args.gain_dbd if args.gain_dbd else 0.0
    loss = args.loss_db if args.loss_db else 0.0
    
    result = core.calculate_erp_info(args.p_tx, gain, loss)
    print(f"\n{'='*50}")
    print(f"ERP / EIRP Calculation")
    print(f"{'='*50}")
    print(result)
    
    return 0


def cmd_db(args):
    """Handle dB calculations."""
    if args.ratio:
        db = core.db_from_ratio(args.ratio)
        print(f"\n{args.ratio}× = {db:.2f} dB")
    elif args.db:
        ratio = core.ratio_from_db(args.db)
        print(f"\n{args.db} dB = {ratio:.4f}×")
    else:
        print("Error: Specify --ratio or --db")
        return 1
    
    return 0


def cmd_muf(args):
    """Handle MUF calculation."""
    if not args.fof2 or not args.distance:
        print("Error: Specify --fof2 and --distance")
        return 1
    
    height = args.height if args.height else 300.0
    result = propagation.calculate_muf(args.fof2, args.distance, height)
    print(f"\n{result}")
    
    # SSZ mode
    if args.ssz is not None:
        # In SSZ mode, effective path length changes
        ssz_result = ssz_extension.compare_skip_with_ssz(args.distance, args.ssz)
        print(f"\n{'='*50}")
        print(f"SSZ Expert Mode (delta_seg = {args.ssz:.4f})")
        print(f"{'='*50}")
        print(f"Classical distance: {ssz_result.classical_value:.0f} km")
        print(f"SSZ eff. distance:  {ssz_result.ssz_value:.0f} km")
        print(f"Difference:         {ssz_result.difference_percent:+.4f}%")
        print(f"\nNote: SSZ modifies effective path length due to")
        print(f"spacetime segmentation effects on propagation.")
    
    return 0


def cmd_field(args):
    """Handle field strength calculation."""
    if not args.p_tx or not args.km:
        print("Error: Specify --p-tx and --km")
        return 1
    
    gain = args.gain_dbi if args.gain_dbi else 0.0
    freq = args.mhz if args.mhz else None
    
    result = field_strength.calculate_field_strength(
        args.p_tx, gain, args.km, freq
    )
    print(f"\n{result}")
    
    return 0


def cmd_ssz(args):
    """Handle SSZ expert mode."""
    if not args.mhz:
        print("Error: Specify --mhz")
        return 1
    
    delta = args.delta if args.delta else 0.01
    
    output = ssz_extension.format_ssz_comparison(args.mhz, delta)
    print(f"\n{output}")
    
    # Show typical values
    if args.info:
        print("\nTypical SSZ values:")
        for name, data in ssz_extension.typical_ssz_values().items():
            print(f"  {name:<20}: δ_seg = {data['delta_seg']:.3f} - {data['description']}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hamtool",
        description="HamTools - Amateur Radio Calculator with SSZ Extension",
        epilog="(c) 2025 Carmen Wrede & Lino Casu",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # === freq command ===
    freq_parser = subparsers.add_parser("freq", help="Frequency/wavelength calculation")
    freq_parser.add_argument("--mhz", type=float, help="Frequency in MHz")
    freq_parser.add_argument("--khz", type=float, help="Frequency in kHz")
    freq_parser.add_argument("--hz", type=float, help="Frequency in Hz")
    freq_parser.add_argument("--ssz", type=float, help="SSZ δ_seg for expert mode")
    
    # === lambda command ===
    lambda_parser = subparsers.add_parser("lambda", help="Wavelength to frequency")
    lambda_parser.add_argument("--m", type=float, help="Wavelength in meters")
    
    # === antenna command ===
    antenna_parser = subparsers.add_parser("antenna", help="Antenna calculations")
    antenna_parser.add_argument("antenna_type", choices=["dipole", "vertical", "yagi"],
                                help="Antenna type")
    antenna_parser.add_argument("--mhz", type=float, help="Frequency in MHz")
    antenna_parser.add_argument("--k", type=float, help="Shortening factor (default 0.95)")
    antenna_parser.add_argument("--elements", type=int, help="Number of elements (Yagi)")
    antenna_parser.add_argument("--boom", type=float, help="Boom length in meters (Yagi)")
    antenna_parser.add_argument("--ssz", type=float, help="SSZ δ_seg for expert mode")
    
    # === feedline command ===
    feedline_parser = subparsers.add_parser("feedline", help="Feedline calculations")
    feedline_parser.add_argument("feedline_cmd", choices=["loss", "compare", "cables"],
                                 help="Feedline subcommand")
    feedline_parser.add_argument("--mhz", type=float, help="Frequency in MHz")
    feedline_parser.add_argument("--cable", type=str, help="Cable type (e.g., RG-58)")
    feedline_parser.add_argument("--length", type=float, help="Cable length in meters")
    feedline_parser.add_argument("--p-in-watt", type=float, help="Input power in Watts")
    
    # === erp command ===
    erp_parser = subparsers.add_parser("erp", help="ERP/EIRP calculation")
    erp_parser.add_argument("--p-tx", type=float, help="TX power in Watts")
    erp_parser.add_argument("--gain-dbd", type=float, help="Antenna gain in dBd")
    erp_parser.add_argument("--loss-db", type=float, help="System losses in dB")
    
    # === db command ===
    db_parser = subparsers.add_parser("db", help="dB calculations")
    db_parser.add_argument("--ratio", type=float, help="Power ratio to convert to dB")
    db_parser.add_argument("--db", type=float, help="dB value to convert to ratio")
    
    # === muf command ===
    muf_parser = subparsers.add_parser("muf", help="MUF calculation")
    muf_parser.add_argument("--fof2", type=float, help="Critical frequency foF2 in MHz")
    muf_parser.add_argument("--distance", type=float, help="Path distance in km")
    muf_parser.add_argument("--height", type=float, help="Virtual height in km (default 300)")
    muf_parser.add_argument("--ssz", type=float, help="SSZ δ_seg for expert mode")
    
    # === field command ===
    field_parser = subparsers.add_parser("field", help="Field strength calculation")
    field_parser.add_argument("--p-tx", type=float, help="TX power in Watts")
    field_parser.add_argument("--gain-dbi", type=float, help="Antenna gain in dBi")
    field_parser.add_argument("--km", type=float, help="Distance in km")
    field_parser.add_argument("--mhz", type=float, help="Frequency in MHz (optional)")
    
    # === ssz command ===
    ssz_parser = subparsers.add_parser("ssz", help="SSZ expert mode")
    ssz_parser.add_argument("--mhz", type=float, help="Frequency in MHz")
    ssz_parser.add_argument("--delta", type=float, help="δ_seg value (default 0.01)")
    ssz_parser.add_argument("--info", action="store_true", help="Show typical SSZ values")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Dispatch to command handlers
    handlers = {
        "freq": cmd_freq,
        "lambda": cmd_lambda,
        "antenna": cmd_antenna,
        "feedline": cmd_feedline,
        "erp": cmd_erp,
        "db": cmd_db,
        "muf": cmd_muf,
        "field": cmd_field,
        "ssz": cmd_ssz,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
