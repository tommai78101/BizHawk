Jump to:
- General
	- [For all development](#for-all-development)
	- [For any: .NET project](#for-any-net-project)
- Projects
	- [blip_buf](#blip_buf)
	- [Cygne](#cygne)
	- [DiscoHawk](#discohawk)
	- [EmuHawk](#emuhawk)
	- [Ext. tools](#ext-tools)
	- [FlatBuffers (managed)](#flatbuffers-managed)
	- [Handy](#handy)
	- [HawkQuantizer](#hawkquantizer)
	- [iso-parser](#iso-parser)
	- [libmupen64plus](#libmupen64plus)
	- [LibretroBridge](#libretrobridge)
	- [LuaInterface](#luainterface)
	- [Mednadisc](#mednadisc)
	- [Misc. submodules](#misc-submodules)
	- [MSXHawk](#msxhawk)
	- [Nyma cores](#nyma-cores)
	- [Octoshock](#octoshock)
	- [QuickNES](#quicknes)
	- [Roslyn Analyzers](#roslyn-analyzers)
	- [Virtu](#virtu)
	- [Waterbox (host)](#waterbox-host)
	- [Waterbox (toolchain + cores)](waterbox-toolchain--cores)
- [Copyrights and licensing](#copyrights-and-licensing)



## For all development

BizHawk's source is hosted with Git.
- Linux
	- If it's not already installed, the package name is probably `git`.
- macOS
	- Git comes with Xcode, or it can be installed manually or with Homebrew ([full instructions](https://git-scm.com/download/mac)).
- Windows
	- Your IDE probably has Git support, or you could install the Git CLI, or a GUI such as [GitHub Desktop](https://desktop.github.com) or [GitKraken](https://gitkraken.com).
	- You can also use Git from within WSL2.

To download the repo, first make sure you have a fork on GitHub (check the dropdown beside the "Fork" button at the top of the page).
Then clone your fork with Git.
You should add the main repo as an extra remote (upstream) to stay up-to-date. With the CLI that would look like:
```sh
git remote add upstream https://github.com/TASEmulators/BizHawk.git
git pull --set-upstream-to=upstream/master master
```

Before touching the code, pull `master` and create a new branch off it with a descriptive name.

After touching the code, commit your changes. Try to group your changes into many smaller commits with a clear purpose to each—committing early and often can help. Bonus points if each commit can build and run.  
If you made the branch a while ago, pull `master` and *rebase, not merge*. Then push to your fork, and you can submit a pull request at any time on GitHub.  
Your commit message summary [should be](https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_commit_guidelines) written in the *imperative* tense (imagine "This commit will" comes before it). GitHub wraps at 70 chars.  
The description should include any non-obvious effects the changes will have. If you feel you need to explain what the code does, consider using code comments instead. It's okay to leave the description blank for simple commits.  
You can use limited Markdown in the summary and description, including monospace, commit/Issue links, and, in the description, bullet points.
In the description, link to related commits and Issues with a short-hash (`abc123def`) or ID (`#1234`), respectively. If your commit fixes an Issue, put it in the summary and use a [closing keyword](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests#linking-a-pull-request-to-an-issue).

It's probably a good idea to get the .NET SDK, even if you're not working on a .NET project, so that you can test your changes in EmuHawk.



## For any: .NET project

- Linux
	- Install the .NET 6 SDK (package name is usually `dotnet-sdk-6.0`, see [full instructions](https://docs.microsoft.com/en-gb/dotnet/core/install/linux)).
	- VS Community isn't available for Linux, but Rider and VS Code are.
- macOS
	- Note that EmuHawk does not currently support macOS.
	- Install the .NET 6 SDK [manually](https://docs.microsoft.com/en-gb/dotnet/core/install/macos) or with Homebrew.
	- VS Community isn't available for macOS, but Rider, VS Code, and VS for Mac are.
- Windows
	- The .NET 6 SDK comes with [VS Community 2022](https://visualstudio.microsoft.com/vs/community) (and it can be manually installed beside VS2019, see [full instructions](https://docs.microsoft.com/en-gb/dotnet/core/install/windows)).
	- You can also use Rider, VS Code, or something else instead of VS Community.

For EmuHawk and libraries in the main solution, which do not target .NET 6, we have [this page](https://github.com/TASEmulators/BizHawk/wiki/Available-C%23-and-.NET-features) documenting which features are actually available to use.



## blip_buf
> Audio resampling library.

Uses C.



## Cygne
> The unmanaged side of the Cygne core from Mednafen.

Uses C++.



## DiscoHawk
> DiscoHawk is a companion app for cleaning up disc images.

See [EmuHawk](#emuhawk). Build scripts also build DiscoHawk, or from VS2022 choose a DiscoHawk configuration.



## EmuHawk
> EmuHawk is the main app, an emulator frontend.

Uses C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).
Most projects target .NET Standard 2.0, with some targeting .NET Framework 4.8. It's written at the top of their project files, or you can check the [project graph](https://gitlab.com/TASVideos/BizHawk/snippets/1886666).

The source for EmuHawk, plus DiscoHawk and the supporting libraries, is in `/src`, with a few extra files used by the build system elsewhere.
EmuHawk's project file `/src/BizHawk.Client.EmuHawk/BizHawk.Client.EmuHawk.csproj` includes the other projects [in a tree](https://gitlab.com/TASVideos/BizHawk/snippets/1886666), and they're all included in `/BizHawk.sln`.

In VS2022, open `BizHawk.sln`, then select the "BizHawk.Client.EmuHawk | Release" configuration to build and run.
On the command-line, from root of the repo run `Dist/BuildRelease.sh` (Unix) or `Dist\QuickTestBuildAndPackage_Release.bat` (Windows). Run EmuHawk from `output` in the repo's root.

There are 2 build configurations. Besides `Release` there is `Debug`, which *does not run* bytecode optimisations, *does not remove* debugging symbols, *enables* additional logging, and *enables* some features. On Windows, a `Debug` executable will spawn a console window for stdout.

We have an automated test suite in the solution which runs in CI, though you can and should run it before pushing.
There are also various Analyzers (static code analysis plugins) for detecting common mistakes and checking code style. Not every style rule is currently enabled, so please make sure you use CRLF, tabs, and [Allman braces](https://en.wikipedia.org/wiki/Indentation_style#Allman_style) (but don't try to fix code you're not working on).

There are additional test suites specifically for regression-testing cores—these are not included in the solution and need to be run manually. See [the base project's readme](https://github.com/TASEmulators/BizHawk/blob/master/src/BizHawk.Tests.Testroms.GB/readme.md) for details.



## Ext. tools
> Various tools/plugins to be loaded in EmuHawk.

These use C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).

### AutoGenConfig
> Experiment in generating settings UI from reflection metadata. (To get something nicer-looking than `PropertyGrid`.)

### DATParser
> Automates gamedb intake.

### DBMan
> Various disc image and gamedb utilities.

### HelloWorld
> Just example code. Not really idiomatic for ext. tools but oh well.



## FlatBuffers (managed)
> Library (incl. generated code) used for serialisation in Nyma cores.

Uses C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).



## Handy
> The unmanaged side of the Handy core from Mednafen.

Uses C++.



## HawkQuantizer
> Library used to encode `.gif` videos.

Uses C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).



## iso-parser
> Library used for disc image parsing.

Uses C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).



## libmupen64plus
> The unmanaged side of the Mupen64Plus core and its bundled plugins.

These use C/C++.



## LibretroBridge
> The unmanaged side of EmuHawk's Libretro adapter.

Uses C++.



## LuaInterface
> Part of EmuHawk's Lua host.

Uses C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).



## Mednadisc
> Library for reading disc images, ported from Mednafen.

Uses C++.



## Misc. submodules

### Emu83
> The unmanaged side of the Emu83 core.

### Gambatte-Speedrun
> The unmanaged side of the Gambatte core.

### libdarm
> Library for ARM disassembly, used for mGBA core.

### libfwunpack
> Library for manipulating NDS firmware data.

### mGBA
> The unmanaged side of the mGBA core.

### SameBoy
> The unmanaged side of the SameBoy core.



## MSXHawk
> The unmanaged side of the MSXHawk core.

Uses C++.



## Nyma cores
> Several cores ported from Mednafen with minimal changes, so tracking upstream is easier.

See [Waterbox (toolchain + cores)](#waterbox-toolchain--cores).



## Octoshock
> The unmanaged side of the Octoshock core from Mednafen.

Uses C++.



## QuickNES
> The unmanaged side of the QuickNES core.

Uses C++.



## Roslyn Analyzers
> Plugins for the C# compiler, used in main BizHawk solution projects.

These use C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).

### BizHawk Analyzer
> Performs additional code style checks not covered by the other Analyzers.

### ReflectionCache source generator
> Generates a helper class in an assembly for accessing `<EmbeddedResource/>`s and `Type`s.



## Virtu
> Library containing most of the Virtu core's emulation code.

Uses C#; you will need the .NET SDK or an IDE which includes it. See the [.NET section](#for-any-net-project).



## Waterbox (host)
> Memory management and other wizardry for loading Waterboxed cores.

Uses Rust.

See [waterboxhost readme](https://github.com/TASEmulators/BizHawk/tree/master/waterbox/waterboxhost#readme).



## Waterbox (toolchain + cores)
> Sandboxing wizardry.

Uses C/C++ targeting Linux w/ a custom libc based on musl.

See [Waterbox readme](https://github.com/TASEmulators/BizHawk/tree/master/waterbox#readme).



## Copyrights and licensing
[//]: # "Changing this section? Don't forget to update the modification date in the PR template!"

By contributing, you declare that any additions or changes either:
- were authored by you (and you are willing to transfer your copyrights to us); or
- were copied from a publicly-licensed source and are properly attributed, including licensing info.
> We will **not** accept any contributions "authored" by GitHub Copilot or similar ML tools.
