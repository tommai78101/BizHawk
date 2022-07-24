﻿using System;
using System.Collections.Generic;

using BizHawk.Emulation.Common;
using BizHawk.Emulation.Cores.Waterbox;

namespace BizHawk.Emulation.Cores.Consoles.Sega.Saturn
{
	[PortedCore(CoreNames.Saturnus, "Mednafen Team", "1.29.0", "https://mednafen.github.io/releases/")]
	public class Saturnus : NymaCore, IRegionable
	{
		private Saturnus(CoreComm comm)
			: base(comm, VSystemID.Raw.NULL, null, null, null)
		{
		}

		private static NymaSettingsInfo _cachedSettingsInfo;

		public static NymaSettingsInfo CachedSettingsInfo(CoreComm comm)
		{
			if (_cachedSettingsInfo is null)
			{
				using var n = new Saturnus(comm);
				n.InitForSettingsInfo("ss.wbx");
				_cachedSettingsInfo = n.SettingsInfo.Clone();
			}

			return _cachedSettingsInfo;
		}

		[CoreConstructor(VSystemID.Raw.SAT)]
		public Saturnus(CoreLoadParameters<NymaSettings, NymaSyncSettings> lp)
			: base(lp.Comm, VSystemID.Raw.SAT, "Saturn Controller", lp.Settings, lp.SyncSettings)
		{
			if (lp.Roms.Count > 0)
				throw new InvalidOperationException("To load a Saturn game, please load the CUE file and not the BIN file.");
			var firmwares = new Dictionary<string, FirmwareID>
			{
				{ "FIRMWARE:$J", new("SAT", "J") },
				{ "FIRMWARE:$U", new("SAT", "U") },
				{ "FIRMWARE:$KOF", new("SAT", "KOF95") },
				{ "FIRMWARE:$ULTRA", new("SAT", "ULTRAMAN") },
				// { "FIRMWARE:$SATAR", new("SAT", "AR") }, // action replay garbage
			};
			DoInit<LibNymaCore>(lp, "ss.wbx", firmwares);

			_cachedSettingsInfo ??= SettingsInfo.Clone();
		}

		protected override IDictionary<string, SettingOverride> SettingOverrides { get; } = new Dictionary<string, SettingOverride>
		{
			{ "ss.bios_jp", new() { Hide = true , Default = "$J" } }, // FIRMWARE:
			{ "ss.bios_na_eu", new() { Hide = true , Default = "$U" } }, // FIRMWARE:
			{ "ss.cart.kof95_path", new() { Hide = true , Default = "$KOF" } }, // FIRMWARE:
			{ "ss.cart.ultraman_path", new() { Hide = true , Default = "$ULTRA" } }, // FIRMWARE:
			{ "ss.cart.satar4mp_path", new() { Hide = true , Default = "$SATAR" } }, // FIRMWARE:
			{ "ss.affinity.vdp2", new() { Hide = true } },
			{ "ss.dbg_exe_cdpath", new() { Hide = true } },
			{ "ss.dbg_exe_cem", new() { Hide = true } },
			{ "ss.dbg_exe_hh", new() { Hide = true } },

			{ "ss.scsp.resamp_quality", new() { NonSync = true } }, // Don't set NoRestart = true for this
			{ "ss.input.mouse_sensitivity", new() { Hide = true } },

			{ "ss.input.port1.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port2.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port3.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port4.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port5.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port6.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port7.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port8.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port9.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port10.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port11.gun_chairs", new() { NonSync = true } },
			{ "ss.input.port12.gun_chairs", new() { NonSync = true } },

			{ "ss.slstart", new() { NonSync = true } },
			{ "ss.slend", new() { NonSync = true } },
			{ "ss.h_overscan", new() { NonSync = true } },
			{ "ss.h_blend", new() { NonSync = true } },
			{ "ss.correct_aspect", new() { NonSync = true, Default = "0" } },
			{ "ss.slstartp", new() { NonSync = true } },
			{ "ss.slendp", new() { NonSync = true } },
		};

		protected override HashSet<string> ComputeHiddenPorts()
		{
			var devCount = 12;
			if (SettingsQuery("ss.input.sport1.multitap") != "1")
				devCount -= 5;
			if (SettingsQuery("ss.input.sport2.multitap") != "1")
				devCount -= 5;
			var ret = new HashSet<string>();
			for (var i = 1; i <= 12; i++)
			{
				if (i > devCount)
					ret.Add($"port{i}");
			}
			return ret;
		}
	}
}
