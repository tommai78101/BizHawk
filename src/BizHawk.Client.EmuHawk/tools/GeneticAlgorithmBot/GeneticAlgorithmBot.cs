using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using BizHawk.Client.Common;
using BizHawk.Common;
using BizHawk.Emulation.Common;
using BizHawk.WinForms.Controls;
using GeneticSharp.Domain;
using GeneticSharp.Domain.Chromosomes;
using GeneticSharp.Domain.Crossovers;
using GeneticSharp.Domain.Fitnesses;
using GeneticSharp.Domain.Mutations;
using GeneticSharp.Domain.Populations;
using GeneticSharp.Domain.Randomizations;
using GeneticSharp.Domain.Selections;
using GeneticSharp.Domain.Terminations;
using Timer = System.Threading.Timer;

/* 
 * Adapted from these sources (all MIT Licensed): 
 * https://github.com/kipgparker/BackPropNetwork/blob/master/BackpropNeuralNetwork/Assets/NeuralNetwork.cs
 * https://github.com/kipgparker/BackPropNetwork/blob/master/LICENSE
 * https://github.com/kipgparker/MutationNetwork/blob/master/Mutation%20Neural%20Network/Assets/NeuralNetwork.cs
 * https://github.com/kipgparker/MutationNetwork/blob/master/LICENSE
 */
namespace BizHawk.Client.EmuHawk
{
	/* 
	 * For example, if given 100 frames, the neural network bot would do something like this: 
	 * 
	 * (input data [11] buttons) -> neural network -> (output data [11] states) 
	 * 
	 * 1 output state = represents how likely it is for the button to be pressed down within those 100 frames. 
	 * 
	 * If the output state is 0.6, there is a 60% chance the button is pressed 1 time in each frame for 100 frames.
	 */
	public sealed partial class GeneticAlgorithmBot : BasicBot
	{
		public static Random rand = new Random((int) DateTime.Now.Ticks);

		private new string _windowTitle = "Genetic Algorithm Bot";
		protected override string WindowTitle => _windowTitle;
		protected override string WindowTitleStatic => _windowTitle;
		public IList<string> ControllerButtons => Emulator.ControllerDefinition.BoolButtons;
		public BotAttempt _bestAttempt => _bestBotAttempt;
		public InputRecording _bestRecording => ((InputRecording) this.ga.BestChromosome);
		public FrameInput[] _lastKnownBestBuffer { get; set; }
		public BotAttempt _gaBestAttempt => ((InputRecording) this.ga.BestChromosome).AttemptAfter;

		private GeneticAlgorithm ga;
		private InputFitnessEvaluator fitnessManager;
		private bool isRunning = false;
		private bool generationIsReady = false;
		private int _gaNumberOfGenerations { get; set; }

		private byte _mainComparisonType { get; set; }
		private byte _tieBreaker1ComparisonType { get; set; }
		private byte _tieBreaker2ComparisonType { get; set; }
		private byte _tieBreaker3ComparisonType { get; set; }

		public GeneticAlgorithmBot() : base()
		{
		}

		public override void StartBot()
		{
			// Check if the Bot is ready for action.
			var message = CanStart();
			if (!string.IsNullOrWhiteSpace(message))
			{
				DialogController.ShowMessageBox(message);
				return;
			}

			// Prepare the UI thread's comparison types
			_mainComparisonType = MainComparisonType;
			_tieBreaker1ComparisonType = Tie1ComparisonType;
			_tieBreaker2ComparisonType = Tie2ComparisonType;
			_tieBreaker3ComparisonType = Tie3ComparisonType;

			_gaNumberOfGenerations = 0;
			this.fitnessManager = new InputFitnessEvaluator(this);
			InputRecording adamChromosome = new InputRecording(this, _startFrame, FrameLength);
			adamChromosome.RandomizeInputRecording();
			adamChromosome.SetBeforeAttempt(MaximizeValue, TieBreaker1Value, TieBreaker2Value, TieBreaker3Value);

			// This operators are classic genetic algorithm operators that lead to a good solution on TSP,
			// but you can try others combinations and see what result you get.
			var crossover = new UniformCrossover();
			var mutation = new UniformMutation();
			var selection = new RouletteWheelSelection();
			var population = new Population(4, 4, adamChromosome);
			this.ga = new GeneticAlgorithm(population, this.fitnessManager, selection, crossover, mutation);
			this.ga.Termination = new GenerationNumberTermination(this._gaNumberOfGenerations + 1);
			this.ga.GenerationRan += (sender, e) =>
			{
				if (!this.generationIsReady)
				{
					Console.WriteLine($"Generation: {this.ga.GenerationsNumber}\t\t{MaximizeValue}");
					this._gaNumberOfGenerations++;
					this.generationIsReady = true;
				}
			};

			base.StartBot();

			if (!this.ga.IsRunning)
			{
				this._bestBotAttempt.Maximize = MaximizeValue;
				this._bestBotAttempt.TieBreak1 = TieBreaker1Value;
				this._bestBotAttempt.TieBreak2 = TieBreaker2Value;
				this._bestBotAttempt.TieBreak3 = TieBreaker3Value;

				this.isRunning = true;
				this.ga.Start();
			}
		}

		public override void StopBot()
		{
			base.StopBot();

			if (this.isRunning)
			{
				this.ga.Stop();
			}
			this.isRunning = false;
		}

		protected override void Update(bool fast)
		{
			if (_doNotUpdateValues)
			{
				return;
			}

			if (!HasFrameAdvanced())
			{
				return;
			}

			if (_replayMode)
			{
				int index = Emulator.Frame - _startFrame;
				this._bestBotAttempt = this._gaBestAttempt;
				if (index < _bestBotAttempt.Log.Count)
				{
					var logEntry = _bestBotAttempt.Log[index];
					var controller = MovieSession.GenerateMovieController();
					controller.SetFromMnemonic(logEntry);
					foreach (var button in controller.Definition.BoolButtons)
					{
						// TODO: make an input adapter specifically for the bot?
						InputManager.ButtonOverrideAdapter.SetButton(button, controller.IsPressed(button));
					}
					InputManager.SyncControls(Emulator, MovieSession, Config);
					_lastFrameAdvanced = Emulator.Frame;
				}
				else
				{
					FinishReplay();
				}
			}
			else if (_isBotting)
			{
				if (Emulator.Frame >= _targetFrame)
				{
					Attempts++;
					Frames += FrameLength;

					PlayBestButton.Enabled = true;

					write_bot_attempts_to_recording();
					if (this._bestBotAttempt.is_Reset || IsBetter(this._bestBotAttempt, this._bestRecording.AttemptAfter))
					{
						Console.WriteLine($"Found best attempt! {this.ga.GenerationsNumber}\t\t{MaximizeValue}");
						copy_recording_to_GA();
						copy_recording_to_last_known_buffer();
						UpdateBestAttemptUI();
					}

					reset_curent(Attempts);
					_doNotUpdateValues = true;
					PressButtons(true);
					MainForm.LoadQuickSave(SelectedSlot, true);
					_lastFrameAdvanced = Emulator.Frame;
					_doNotUpdateValues = false;

					if (this.isRunning)
					{
						this.generationIsReady = false;
						this.ga.Termination = new GenerationNumberTermination(this._gaNumberOfGenerations + 1);
						this.ga.Resume();
					}

					return;
				}

				// Before this would have 2 additional hits before the frame even advanced, making the amount of inputs greater than the number of frames to test.
				//aka do not Add more inputs than there are Frames to test
				if (this._gaBestAttempt.Log.Count < FrameLength)
				{
					PressButtons(false);
					_lastFrameAdvanced = Emulator.Frame;
					copy_recording_to_GA();
				}
			}
		}

		protected override void PressButtons(bool clear_log)
		{
			if (this._bestRecording != null)
			{
				FrameInput inputs = this._bestRecording.GetFrameInput(Emulator.Frame);
				foreach (var button in inputs.Buttons)
				{
					InputManager.ClickyVirtualPadController.SetBool(button, false);
				}
				InputManager.SyncControls(Emulator, MovieSession, Config);

				if (clear_log) { _gaBestAttempt.Log.Clear(); }
				_gaBestAttempt.Log.Add(_logGenerator.GenerateLogEntry());
			}
		}

		public void UpdateBestAttemptUI()
		{
			Invoke((MethodInvoker) (() =>
			{
				copy_GA_to_best();
				UpdateBestAttempt();
			}));
		}

		public float[] GetCachedInputProbabilities()
		{
			float[] target = new float[Emulator.ControllerDefinition.BoolButtons.Count];
			for (int i = 0; i < Emulator.ControllerDefinition.BoolButtons.Count; i++)
			{
				string button = Emulator.ControllerDefinition.BoolButtons[i];
				target[i] = (float) ControlProbabilities[button] / 100.0f;
			}
			return target;
		}

		public double GetBetterValue(BotAttempt comparison, BotAttempt current)
		{
			static bool TestValue(byte operation, int currentValue, int bestValue)
				=> operation switch
				{
					0 => (currentValue > bestValue),
					1 => (currentValue >= bestValue),
					2 => (currentValue == bestValue),
					3 => (currentValue <= bestValue),
					4 => (currentValue < bestValue),
					5 => (currentValue != bestValue),
					_ => false
				};

			if (TestValue(MainComparisonType, current.Maximize, comparison.Maximize)) return comparison.Maximize - current.Maximize;
			if (TestValue(Tie1ComparisonType, current.TieBreak1, comparison.TieBreak1)) return comparison.TieBreak1 - current.TieBreak1;
			if (TestValue(Tie2ComparisonType, current.TieBreak2, comparison.TieBreak2)) return comparison.TieBreak2 - current.TieBreak2;
			return comparison.TieBreak3 - current.TieBreak3;
		}

		public void CopyBuffer(out FrameInput[] target)
		{
			int length = this._lastKnownBestBuffer?.Length ?? this.FrameLength;
			target = new FrameInput[length];
			if (this._lastKnownBestBuffer != null && this._lastKnownBestBuffer.Length > 0)
			{
				// Only copy the buffer up to the lowest common denominator length.
				for (int i = 0; i < length; i++)
				{
					target[i] = this._lastKnownBestBuffer[i];
				}
			}
		}

		public void copy_GA_to_best()
		{
			this._bestBotAttempt.Attempt = this._gaBestAttempt.Attempt;
			this._bestBotAttempt.Maximize = this._gaBestAttempt.Maximize;
			this._bestBotAttempt.TieBreak1 = this._gaBestAttempt.TieBreak1;
			this._bestBotAttempt.TieBreak2 = this._gaBestAttempt.TieBreak2;
			this._bestBotAttempt.TieBreak3 = this._gaBestAttempt.TieBreak3;

			this._bestBotAttempt.Log.Clear();
			for (int i = 0; i < this._gaBestAttempt.Log.Count; i++)
			{
				this._bestBotAttempt.Log.Add(this._gaBestAttempt.Log[i]);
			}
			this._bestBotAttempt.is_Reset = false;
		}

		private void copy_recording_to_GA()
		{
			this._gaBestAttempt.Attempt = this._bestRecording.AttemptAfter.Attempt;
			this._gaBestAttempt.Maximize = this._bestRecording.AttemptAfter.Maximize;
			this._gaBestAttempt.TieBreak1 = this._bestRecording.AttemptAfter.TieBreak1;
			this._gaBestAttempt.TieBreak2 = this._bestRecording.AttemptAfter.TieBreak2;
			this._gaBestAttempt.TieBreak3 = this._bestRecording.AttemptAfter.TieBreak3;
		}

		private void copy_recording_to_last_known_buffer()
		{
			this._lastKnownBestBuffer = new FrameInput[this._bestRecording.InputBuffer.Length];
			this._bestRecording.InputBuffer.CopyTo(this._lastKnownBestBuffer, 0);
		}

		private void write_bot_attempts_to_recording()
		{
			this._bestRecording.SetAfterAttempt(MaximizeValue, TieBreaker1Value, TieBreaker2Value, TieBreaker3Value);
		}
	}


	public class InputFitnessEvaluator : IFitness
	{
		private GeneticAlgorithmBot bot;

		public ClickyVirtualPadController BotController => bot.InputManager.ClickyVirtualPadController;
		public IList<string> ControllerButtons => bot.ControllerButtons;
		public InputManager InputManager => bot.InputManager;

		public InputFitnessEvaluator(GeneticAlgorithmBot owner)
		{
			this.bot = owner;
		}

		public double Evaluate(IChromosome chromosome)
		{
			double distanceFromTargetValue = 0.0;
			double fitness = double.Epsilon;

			if (bot._bestRecording != null)
			{
				bot._bestRecording.AttemptAfter.is_Reset = false;
				if (bot.IsBetter(bot._bestBotAttempt, bot._gaBestAttempt))
				{
					distanceFromTargetValue = bot.GetBetterValue(bot._bestAttempt, bot._bestRecording.AttemptAfter);
					if (distanceFromTargetValue <= double.Epsilon)
						distanceFromTargetValue = double.Epsilon;
					fitness = (1.0 / (distanceFromTargetValue * distanceFromTargetValue));
				}
			}

			if (fitness < 0.0)
			{
				fitness *= -1.0;
			}
			if (fitness < double.Epsilon || double.IsInfinity(fitness) || double.IsNaN(fitness))
			{
				fitness = double.Epsilon;
			}
			return fitness;
		}
	}

	public class InputRecording : ChromosomeBase
	{
		public int StartFrameNumber { get; set; }
		public FrameInput[] InputBuffer { get; set; }
		public GeneticAlgorithmBot bot;
		public ClickyVirtualPadController BotController => bot.InputManager.ClickyVirtualPadController;
		public IList<string> ControllerButtons => bot.ControllerButtons;
		public InputManager InputManager => bot.InputManager;
		public BotAttempt AttemptBefore { get; set; }
		public BotAttempt AttemptAfter { get; set; }

		// At most, a game theoretically can only enter up to this many amount of buttons pressed simulataneously on a single frame.
		// This is the same with standard keyboard rollover only supporting up to 6 keys.
		public readonly int SimultaneousInputSize = 6;

		public InputRecording(GeneticAlgorithmBot owner, int startFrame, int frameLength) : base(frameLength)
		{
			this.bot = owner;
			this.StartFrameNumber = startFrame;
			this.InputBuffer = new FrameInput[frameLength];
			for (int i = 0; i < frameLength; i++)
			{
				this.InputBuffer[i] = new FrameInput(startFrame + i);
			}
			this.AttemptBefore = new BotAttempt();
			this.AttemptAfter = new BotAttempt();
			this.AttemptBefore.is_Reset = true;
			this.AttemptAfter.is_Reset = true;
			CreateGenes();
		}

		public void SetBeforeAttempt(int max, int tie1, int tie2, int tie3)
		{
			this.AttemptBefore.Maximize = max;
			this.AttemptBefore.TieBreak1 = tie1;
			this.AttemptBefore.TieBreak2 = tie2;
			this.AttemptBefore.TieBreak3 = tie3;
			this.AttemptBefore.is_Reset = false;
		}

		public void SetAfterAttempt(int max, int tie1, int tie2, int tie3)
		{
			this.AttemptAfter.Maximize = max;
			this.AttemptAfter.TieBreak1 = tie1;
			this.AttemptAfter.TieBreak2 = tie2;
			this.AttemptAfter.TieBreak3 = tie3;
			this.AttemptAfter.is_Reset = false;
		}

		public void SetAfterAttempt(BotAttempt value)
		{
			this.AttemptAfter.Maximize = value.Maximize;
			this.AttemptAfter.TieBreak1 = value.TieBreak1;
			this.AttemptAfter.TieBreak2 = value.TieBreak2;
			this.AttemptAfter.TieBreak3 = value.TieBreak3;
		}

		public FrameInput GetFrameInput(int frameNumber)
		{
			int index = frameNumber - this.StartFrameNumber;
			if (index < 0 || index >= this.InputBuffer.Length)
			{
				index = this.InputBuffer.Length - 1;
			}
			return this.InputBuffer[index];
		}

		public void SetFrameInput(int index, FrameInput input)
		{
			HashSet<string> copy = new HashSet<string>();
			copy.UnionWith(input.Buttons);
			if (0 <= index && index < this.InputBuffer.Length)
			{
				this.InputBuffer[index].Buttons.Clear();
				this.InputBuffer[index].Buttons.UnionWith(copy);
			}
		}

		public void RandomizeInputRecording()
		{
			float[] probabilities = bot.GetCachedInputProbabilities();
			IList<int[]> a = Enumerable.Range(0, this.bot.FrameLength).Select(run =>
			{
				int[] times = Enumerable.Range(0, ControllerButtons.Count)
					.Where((buttonIndex, i) => RandomizationProvider.Current.GetDouble() < probabilities[buttonIndex])
					.ToArray();
				return times;
			}).ToArray();
			int[][] values = a.ToArray();

			int length = values.Length;
			if (values.Length != this.bot.FrameLength)
			{
				length = this.bot.FrameLength;
			}

			for (int i = 0; i < length; i++)
			{
				FrameInput input = this.GetFrameInput(this.StartFrameNumber + i);
				for (int j = 0; j < values[i].Length; j++)
				{
					input.Pressed(ControllerButtons[values[i][j]]);
				}
			}
		}

		public void RandomizeFrameInput()
		{
			int frameNumber = RandomizationProvider.Current.GetInt(bot._startFrame, bot._startFrame + this.InputBuffer.Length);
			int index = frameNumber - bot._startFrame;
			FrameInput input = this.GetFrameInput(frameNumber);
			input.Clear();

			float[] probabilities = bot.GetCachedInputProbabilities();
			int[] times = Enumerable.Range(0, count: ControllerButtons.Count)
					.Where((buttonIndex, i) => RandomizationProvider.Current.GetDouble() < probabilities[buttonIndex])
					.ToArray();

			for (int i = 0; i < times.Length; i++)
			{
				input.Pressed(ControllerButtons[times[i]]);
			}
		}

		public override Gene GenerateGene(int geneIndex)
		{
			return new Gene(RandomizationProvider.Current.GetInt(0, this.InputBuffer.Length));
		}

		public override IChromosome CreateNew()
		{
			InputRecording copy = new InputRecording(this.bot, this.bot._startFrame, this.InputBuffer.Length);
			if (this.bot._lastKnownBestBuffer != null)
			{
				FrameInput[] target = new FrameInput[this.bot._lastKnownBestBuffer?.Length ?? this.bot.FrameLength];
				this.bot.CopyBuffer(out target);
				target.CopyTo(copy.InputBuffer, 0);
			}
			else
			{
				this.InputBuffer.CopyTo(copy.InputBuffer, 0);
			}
			copy.RandomizeFrameInput();
			return copy;
		}

		public override IChromosome Clone()
		{
			InputRecording copy = base.Clone() as InputRecording;
			copy.bot = this.bot;
			copy.InputBuffer = new FrameInput[this.InputBuffer.Length];
			Array.Copy(this.InputBuffer, copy.InputBuffer, this.InputBuffer.Length);
			return copy;
		}
	}

	public class FrameInput
	{
		public HashSet<string> Buttons { get; set; }
		public int FrameNumber { get; set; }

		public FrameInput(int frameNumber)
		{
			this.Buttons = new HashSet<string>();
			FrameNumber = frameNumber;
		}

		public void Clear()
		{
			this.Buttons.Clear();
		}

		public void Pressed(string button)
		{
			this.Buttons.Add(button);
		}

		public void Released(string button)
		{
			this.Buttons.Remove(button);
		}

		public bool IsPressed(string button)
		{
			return this.Buttons.Contains(button);
		}
	}
}

