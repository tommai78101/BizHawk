﻿using System;

using BizHawk.Emulation.Common;

namespace BizHawk.Emulation.Cores.Nintendo.Gameboy
{
	public partial class Gameboy : ISaveRam
	{
		public bool SaveRamModified
		{
			get
			{
				if (LibGambatte.gambatte_getsavedatalength(GambatteState) == 0)
				{
					return false;
				}

				return true; // need to wire more stuff into the core to actually know this
			}
		}

		public byte[] CloneSaveRam()
		{
			int length = LibGambatte.gambatte_getsavedatalength(GambatteState);

			if (length > 0)
			{
				byte[] ret = new byte[length];
				LibGambatte.gambatte_savesavedata(GambatteState, ret);
				return ret;
			}

			return new byte[0];
		}

		public void StoreSaveRam(byte[] data)
		{
			int expected = LibGambatte.gambatte_getsavedatalength(GambatteState);
			if (data.Length != expected) throw new ArgumentException(message: "Size of saveram data does not match expected!", paramName: nameof(data));

			LibGambatte.gambatte_loadsavedata(GambatteState, data);

			if (DeterministicEmulation)
			{
				ulong dividers = _syncSettings.InitialTime * (0x400000UL + (ulong)_syncSettings.RTCDivisorOffset) / 2UL;
				LibGambatte.gambatte_settime(GambatteState, dividers);
			}
		}
	}
}
