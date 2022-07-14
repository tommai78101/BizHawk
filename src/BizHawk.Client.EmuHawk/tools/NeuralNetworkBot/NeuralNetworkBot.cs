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
using GeneticSharp.Infrastructure.Framework.Threading;

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
	public sealed partial class NeuralNetworkBot : BasicBot, IFitness
	{
		public static Random rand = new Random((int) DateTime.Now.Ticks);

		private new string _windowTitle = "Neural Network Bot";

		protected override string WindowTitle => _windowTitle;
		protected override string WindowTitleStatic => _windowTitle;

		public static float MutationChance { get; } = 0.01f;
		public static float MutationStrength { get; } = 0.5f;

		public static int NeuralNetworkPoolSize { get; } = 10;

		private GeneticAlgorithm ga;
		private Thread gaThread;
		private bool IsGaThreadRunning { get; set; }

		private byte _mainComparisonType { get; set; }
		private byte _tieBreaker1ComparisonType { get; set; }
		private byte _tieBreaker2ComparisonType { get; set; }
		private byte _tieBreaker3ComparisonType { get; set; }


		protected NeuralNetwork neuralNetwork { get; set; }
		protected NeuralNetwork bestNeuralNetwork { get; set; }
		protected IList<NeuralNetwork> Networks { get; set; }
		protected BotAttempt FitnessComparison { get; set; }

		public static float[] FeedForwards;

		public NeuralNetworkBot() : base()
		{
			FitnessComparison = new BotAttempt();
			FitnessComparison.is_Reset = true;
		}

		// Overriding this function, because we need to insert neural network code in this function invocation.
		protected override void PressButtons(bool clear_log)
		{
			// Set-up for the controller bias.
			float[] inputs = GetCachedInputProbabilities();
			for (int i = 0; i < Emulator.ControllerDefinition.BoolButtons.Count; i++)
			{
				float value = FeedForwards[i];
				if (float.IsNaN(value) || float.IsInfinity(value))
				{
					// Need to continue improving this bot.
					StopBot();
					return;
				}
				string button = Emulator.ControllerDefinition.BoolButtons[i];
				bool pressed = CalculateProbability(value);
				InputManager.ClickyVirtualPadController.SetBool(button, pressed);
			}
			InputManager.SyncControls(Emulator, MovieSession, Config);

			if (clear_log)
			{
				this.bestNeuralNetwork.BotFitness.Log.Clear();
			}
			this.bestNeuralNetwork.BotFitness.Log.Add(_logGenerator.GenerateLogEntry());
		}

		private bool CalculateProbability(float probability)
		{
			// "False" means "pressed down".
			return !((rand.NextDouble() * 100) <= probability);
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

		private void Scramble(out float[] target)
		{
			target = new float[Emulator.ControllerDefinition.BoolButtons.Count];
			for (int i = 0; i < Emulator.ControllerDefinition.BoolButtons.Count; i++)
			{
				target[i] = (float) rand.NextDouble();
			}
		}

		protected override bool IsBetter(BotAttempt comparison, BotAttempt current)
		{
			if (!TestValue(_mainComparisonType, current.Maximize, comparison.Maximize))
			{
				return false;
			}

			if (current.Maximize == comparison.Maximize)
			{
				if (!TestValue(_tieBreaker1ComparisonType, current.TieBreak1, comparison.TieBreak1))
				{
					return false;
				}

				if (current.TieBreak1 == comparison.TieBreak1)
				{
					if (!TestValue(_tieBreaker2ComparisonType, current.TieBreak2, comparison.TieBreak2))
					{
						return false;
					}

					if (current.TieBreak2 == comparison.TieBreak2)
					{
						if (!TestValue(_tieBreaker3ComparisonType, current.TieBreak3, current.TieBreak3))
						{
							return false;
						}
					}
				}
			}

			return true;
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

			// Prepare the multi-layer perceptron management.
			int inputSize = Emulator.ControllerDefinition.BoolButtons.Count;
			int hiddenLayerSize = (int) Math.Round((inputSize * 1.66f), 0);
			int[] layersInfo = new int[] { inputSize, hiddenLayerSize, hiddenLayerSize, inputSize };
			int[] activations = new int[] { 2, 2, 2, 2 };
			FeedForwards = new float[inputSize];
			Array.Clear(FeedForwards, 0, FeedForwards.Length);

			this.Networks = new List<NeuralNetwork>(NeuralNetworkPoolSize);
			for (int i = 0; i < NeuralNetworkPoolSize; i++)
			{
				this.Networks.Add(new NeuralNetwork(this, layersInfo, activations));
			}
			this.bestNeuralNetwork = new NeuralNetwork(this, layersInfo, activations);
			this.neuralNetwork = new NeuralNetwork(this, layersInfo, activations);

			this.bestNeuralNetwork.InitializeNeuralNetwork(layersInfo, activations);

			#region
			/*
			this.feedForwards = new float[inputSize];
			for (int i = 0; i < inputSize; i++)
			{
				this.feedForwards[i] = (float) rand.NextDouble();
			}

			if (this.bestNeuralNetwork == null)
			{
				this.neuralNetwork = new NeuralNetwork(new int[] { inputSize, hiddenLayerSize, hiddenLayerSize, inputSize }, new string[] { "leakyrelu", "leakyrelu", "leakyrelu", "leakyrelu" });
				this.bestNeuralNetwork = new NeuralNetwork(this.neuralNetwork);
			}
			else
			{
				this.neuralNetwork = new NeuralNetwork(this.bestNeuralNetwork);
			}

			// Begin training data
			this.bestNeuralNetwork.Fitness.is_Reset = true;
			bool[] bitArray = new bool[inputSize];
			float[] trainingInputs = new float[inputSize];
			float[] trainingResults = new float[inputSize];

			for (int i = 0; i < inputSize; i++)
			{
				string button = Emulator.ControllerDefinition.BoolButtons[i];
				trainingResults[i] = (float) ControlProbabilities[button];
			}

			for (int i = 0; i < inputSize * inputSize * inputSize; i++)
			{
				neuralNetwork.BackPropagate(trainingInputs, trainingResults);
				for (int b = 0; b < inputSize && !(bitArray[b] = !bitArray[b++]);) { }
				for (int b = 0; b < inputSize; trainingInputs[b] = (bitArray[b++] ? 1f : 0f)) { }
			}
			*/
			#endregion


			base.StartBot();

			IFitness fitness = this;
			IChromosome chromosome = new NeuralNetwork(this, layersInfo, activations);
			// This operators are classic genetic algorithm operators that lead to a good solution on TSP,
			// but you can try others combinations and see what result you get.
			var crossover = new UniformCrossover();
			var mutation = new ReverseSequenceMutation();
			var selection = new RouletteWheelSelection();
			var population = new Population(10, 10, chromosome);

			ga = new GeneticAlgorithm(population, fitness, selection, crossover, mutation);
			ga.Termination = new FitnessStagnationTermination(100);
			ga.TaskExecutor = new ParallelTaskExecutor
			{
				MinThreads = 2,
				MaxThreads = 4
			};
			ga.GenerationRan += new EventHandler((o, i) =>
			{
				this.bestNeuralNetwork = (NeuralNetwork) ga.BestChromosome;
				BotAttempt attempt = this.bestNeuralNetwork.BotFitness;
				attempt.Attempt = ga.GenerationsNumber;
				UpdateBestAttempt(attempt);
				FeedForwards = this.bestNeuralNetwork.ForwardPropagation(FeedForwards);
			});

			gaThread = new Thread(() => ga.Start());
			gaThread.Name = "Genetic Algorithm Thread";

			if (!ga.IsRunning)
			{
				gaThread.Start();
			}
		}

		public override void StopBot()
		{
			base.StopBot();

			if (ga.IsRunning)
			{
				ga.Stop();
			}

			//this.neuralNetwork.Fitness.is_Reset = true;
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
				if (index < this.bestNeuralNetwork.BotFitness.Log.Count)
				{
					var logEntry = this.bestNeuralNetwork.BotFitness.Log[index];
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

					//this.neuralNetwork.Fitness.Maximize = MaximizeValue;
					//this.neuralNetwork.Fitness.TieBreak1 = TieBreaker1Value;
					//this.neuralNetwork.Fitness.TieBreak2 = TieBreaker2Value;
					//this.neuralNetwork.Fitness.TieBreak3 = TieBreaker3Value;
					PlayBestButton.Enabled = true;

					//if (this.bestNeuralNetwork.Fitness.is_Reset || IsBetter(this.bestNeuralNetwork.Fitness, this.neuralNetwork.Fitness))
					//{
					//	copy_curent_to_best();
					//	UpdateBestAttempt();
					//}

					reset_curent(Attempts);
					_doNotUpdateValues = true;
					PressButtons(true);
					MainForm.LoadQuickSave(SelectedSlot, true);
					_lastFrameAdvanced = Emulator.Frame;
					_doNotUpdateValues = false;

					// Neural network update here

					//bestNeuralNetwork.Mutate((int) (1.0f / MutationChance), MutationStrength);

					return;
				}

				// Before this would have 2 additional hits before the frame even advanced, making the amount of inputs greater than the number of frames to test.
				if (this.neuralNetwork.BotFitness.Log.Count < FrameLength) //aka do not Add more inputs than there are Frames to test
				{
					PressButtons(false);
					_lastFrameAdvanced = Emulator.Frame;
				}
			}
		}

		protected override void UpdateBestAttempt()
		{
			if (!this.bestNeuralNetwork.BotFitness.is_Reset)
			{
				ClearBestButton.Enabled = true;
				BestAttemptNumberLabel.Text = this.bestNeuralNetwork.BotFitness.Attempt.ToString();
				BestMaximizeBox.Text = this.bestNeuralNetwork.BotFitness.Maximize.ToString();
				BestTieBreak1Box.Text = this.bestNeuralNetwork.BotFitness.TieBreak1.ToString();
				BestTieBreak2Box.Text = this.bestNeuralNetwork.BotFitness.TieBreak2.ToString();
				BestTieBreak3Box.Text = this.bestNeuralNetwork.BotFitness.TieBreak3.ToString();

				var sb = new StringBuilder();
				foreach (var logEntry in this.bestNeuralNetwork.BotFitness.Log)
				{
					sb.AppendLine(logEntry);
				}
				BestAttemptLogLabel.Text = sb.ToString();
				PlayBestButton.Enabled = true;
			}
			else
			{
				ClearBestButton.Enabled = false;
				BestAttemptNumberLabel.Text = "";
				BestMaximizeBox.Text = "";
				BestTieBreak1Box.Text = "";
				BestTieBreak2Box.Text = "";
				BestTieBreak3Box.Text = "";
				BestAttemptLogLabel.Text = "";
				PlayBestButton.Enabled = false;
			}
		}

		private void UpdateBestAttempt(BotAttempt attempt)
		{
			Invoke(new MethodInvoker(() =>
			{
				if (!attempt.is_Reset)
				{
					ClearBestButton.Enabled = true;
					BestAttemptNumberLabel.Text = attempt.Attempt.ToString();
					BestMaximizeBox.Text = attempt.Maximize.ToString();
					BestTieBreak1Box.Text = attempt.TieBreak1.ToString();
					BestTieBreak2Box.Text = attempt.TieBreak2.ToString();
					BestTieBreak3Box.Text = attempt.TieBreak3.ToString();

					var sb = new StringBuilder();
					foreach (var logEntry in attempt.Log)
					{
						sb.AppendLine(logEntry);
					}
					BestAttemptLogLabel.Text = sb.ToString();
					PlayBestButton.Enabled = true;
				}
				else
				{
					ClearBestButton.Enabled = false;
					BestAttemptNumberLabel.Text = "";
					BestMaximizeBox.Text = "";
					BestTieBreak1Box.Text = "";
					BestTieBreak2Box.Text = "";
					BestTieBreak3Box.Text = "";
					BestAttemptLogLabel.Text = "";
					PlayBestButton.Enabled = false;
				}
			}));
		}

		protected override void UpdateComparisonBotAttempt()
		{
			if (this.bestNeuralNetwork.BotFitness.is_Reset)
			{
				if (MainBestRadio.Checked)
				{
					FitnessComparison.Maximize = 0;
				}

				if (TieBreak1BestRadio.Checked)
				{
					FitnessComparison.TieBreak1 = 0;
				}

				if (TieBreak2BestRadio.Checked)
				{
					FitnessComparison.TieBreak2 = 0;
				}

				if (TieBreak3BestRadio.Checked)
				{
					FitnessComparison.TieBreak3 = 0;
				}
			}
			else
			{
				if (MainBestRadio.Checked && this.bestNeuralNetwork.BotFitness.Maximize != FitnessComparison.Maximize)
				{
					FitnessComparison.Maximize = this.bestNeuralNetwork.BotFitness.Maximize;
				}

				if (TieBreak1BestRadio.Checked && this.bestNeuralNetwork.BotFitness.TieBreak1 != FitnessComparison.TieBreak1)
				{
					FitnessComparison.TieBreak1 = this.bestNeuralNetwork.BotFitness.TieBreak1;
				}

				if (TieBreak2BestRadio.Checked && this.bestNeuralNetwork.BotFitness.TieBreak2 != FitnessComparison.TieBreak2)
				{
					FitnessComparison.TieBreak2 = this.bestNeuralNetwork.BotFitness.TieBreak2;
				}

				if (TieBreak3BestRadio.Checked && this.bestNeuralNetwork.BotFitness.TieBreak3 != FitnessComparison.TieBreak3)
				{
					FitnessComparison.TieBreak3 = this.bestNeuralNetwork.BotFitness.TieBreak3;
				}
			}
		}

		protected override void reset_curent(long attempt_num)
		{
			this.neuralNetwork.BotFitness.Attempt = attempt_num;
			this.neuralNetwork.BotFitness.Maximize = 0;
			this.neuralNetwork.BotFitness.TieBreak1 = 0;
			this.neuralNetwork.BotFitness.TieBreak2 = 0;
			this.neuralNetwork.BotFitness.TieBreak3 = 0;
			this.neuralNetwork.BotFitness.Log.Clear();
			this.neuralNetwork.BotFitness.is_Reset = true;
		}

		protected override void copy_curent_to_best()
		{
			this.bestNeuralNetwork.BotFitness.Attempt = this.neuralNetwork.BotFitness.Attempt;
			this.bestNeuralNetwork.BotFitness.Maximize = this.neuralNetwork.BotFitness.Maximize;
			this.bestNeuralNetwork.BotFitness.TieBreak1 = this.neuralNetwork.BotFitness.TieBreak1;
			this.bestNeuralNetwork.BotFitness.TieBreak2 = this.neuralNetwork.BotFitness.TieBreak2;
			this.bestNeuralNetwork.BotFitness.TieBreak3 = this.neuralNetwork.BotFitness.TieBreak3;

			// no references to ComparisonType parameters

			this.bestNeuralNetwork.BotFitness.Log.Clear();

			for (int i = 0; i < this.neuralNetwork.BotFitness.Log.Count; i++)
			{
				this.bestNeuralNetwork.BotFitness.Log.Add(this.neuralNetwork.BotFitness.Log[i]);
			}

			this.bestNeuralNetwork.BotFitness.is_Reset = false;
		}

		protected override bool LoadBotFile(string path)
		{
			var file = new FileInfo(path);
			if (!file.Exists)
			{
				return false;
			}

			var json = File.ReadAllText(path);
			var botData = (NeuralNetworkBotData) ConfigService.LoadWithType(json);

			this.bestNeuralNetwork = new NeuralNetwork(this, botData.layersInfo, botData.biases, botData.neurons, botData.weights, botData.activations);
			this.bestNeuralNetwork.BotFitness.Attempt = botData.Best.Attempt;
			this.bestNeuralNetwork.BotFitness.Maximize = botData.Best.Maximize;
			this.bestNeuralNetwork.BotFitness.TieBreak1 = botData.Best.TieBreak1;
			this.bestNeuralNetwork.BotFitness.TieBreak2 = botData.Best.TieBreak2;
			this.bestNeuralNetwork.BotFitness.TieBreak3 = botData.Best.TieBreak3;

			// no references to ComparisonType parameters

			this.bestNeuralNetwork.BotFitness.Log.Clear();

			for (int i = 0; i < botData.Best.Log.Count; i++)
			{
				this.bestNeuralNetwork.BotFitness.Log.Add(botData.Best.Log[i]);
			}

			this.bestNeuralNetwork.BotFitness.is_Reset = false;

			var probabilityControls = ControlProbabilityPanel.Controls
					.OfType<BotControlsRow>()
					.ToList();

			foreach (var (button, p) in botData.ControlProbabilities)
			{
				var control = probabilityControls.Single(c => c.ButtonName == button);
				control.Probability = p;
			}

			MaximizeAddress = botData.Maximize;
			TieBreaker1Address = botData.TieBreaker1;
			TieBreaker2Address = botData.TieBreaker2;
			TieBreaker3Address = botData.TieBreaker3;
			try
			{
				MainComparisonType = botData.ComparisonTypeMain;
				Tie1ComparisonType = botData.ComparisonTypeTie1;
				Tie2ComparisonType = botData.ComparisonTypeTie2;
				Tie3ComparisonType = botData.ComparisonTypeTie3;

				MainBestRadio.Checked = botData.MainCompareToBest;
				TieBreak1BestRadio.Checked = botData.TieBreaker1CompareToBest;
				TieBreak2BestRadio.Checked = botData.TieBreaker2CompareToBest;
				TieBreak3BestRadio.Checked = botData.TieBreaker3CompareToBest;
				MainValueRadio.Checked = !botData.MainCompareToBest;
				TieBreak1ValueRadio.Checked = !botData.TieBreaker1CompareToBest;
				TieBreak2ValueRadio.Checked = !botData.TieBreaker2CompareToBest;
				TieBreak3ValueRadio.Checked = !botData.TieBreaker3CompareToBest;

				MainValueNumeric.Value = botData.MainCompareToValue;
				TieBreak1Numeric.Value = botData.TieBreaker1CompareToValue;
				TieBreak2Numeric.Value = botData.TieBreaker2CompareToValue;
				TieBreak3Numeric.Value = botData.TieBreaker3CompareToValue;
			}
			catch
			{
				MainComparisonType = 0;
				Tie1ComparisonType = 0;
				Tie2ComparisonType = 0;
				Tie3ComparisonType = 0;

				MainBestRadio.Checked = true;
				TieBreak1BestRadio.Checked = true;
				TieBreak2BestRadio.Checked = true;
				TieBreak3BestRadio.Checked = true;
				MainBestRadio.Checked = false;
				TieBreak1BestRadio.Checked = false;
				TieBreak2BestRadio.Checked = false;
				TieBreak3BestRadio.Checked = false;

				MainValueNumeric.Value = 0;
				TieBreak1Numeric.Value = 0;
				TieBreak2Numeric.Value = 0;
				TieBreak3Numeric.Value = 0;
			}
			FrameLength = botData.FrameLength;
			FromSlot = botData.FromSlot;
			Attempts = botData.Attempts;
			Frames = botData.Frames;

			_currentDomain = !string.IsNullOrWhiteSpace(botData.MemoryDomain)
					? MemoryDomains[botData.MemoryDomain]
					: MemoryDomains.MainMemory;

			_bigEndian = botData.BigEndian;
			_dataSize = botData.DataSize > 0 ? botData.DataSize : 1;

			UpdateBestAttempt();
			UpdateComparisonBotAttempt();

			if (!this.bestNeuralNetwork.BotFitness.is_Reset)
			{
				PlayBestButton.Enabled = true;
			}

			CurrentFileName = path;
			Settings.RecentBotFiles.Add(CurrentFileName);
			MessageLabel.Text = $"{Path.GetFileNameWithoutExtension(path)} loaded";

			AssessRunButtonStatus();
			return true;
		}

		protected override void SaveBotFile(string path)
		{
			var data = new NeuralNetworkBotData
			{
				Best = this.bestNeuralNetwork?.BotFitness ?? new BotAttempt(),
				ControlProbabilities = ControlProbabilities,
				Maximize = MaximizeAddress,
				TieBreaker1 = TieBreaker1Address,
				TieBreaker2 = TieBreaker2Address,
				TieBreaker3 = TieBreaker3Address,
				ComparisonTypeMain = MainComparisonType,
				ComparisonTypeTie1 = Tie1ComparisonType,
				ComparisonTypeTie2 = Tie2ComparisonType,
				ComparisonTypeTie3 = Tie3ComparisonType,
				MainCompareToBest = MainBestRadio.Checked,
				TieBreaker1CompareToBest = TieBreak1BestRadio.Checked,
				TieBreaker2CompareToBest = TieBreak2BestRadio.Checked,
				TieBreaker3CompareToBest = TieBreak3BestRadio.Checked,
				MainCompareToValue = (int) MainValueNumeric.Value,
				TieBreaker1CompareToValue = (int) TieBreak1Numeric.Value,
				TieBreaker2CompareToValue = (int) TieBreak2Numeric.Value,
				TieBreaker3CompareToValue = (int) TieBreak3Numeric.Value,
				FromSlot = FromSlot,
				FrameLength = FrameLength,
				Attempts = Attempts,
				Frames = Frames,
				MemoryDomain = _currentDomain.Name,
				BigEndian = _bigEndian,
				DataSize = _dataSize,
			};

			// Deep copying arrays
			float[] inputs = GetCachedInputProbabilities();
			int hiddenSize = (int) Math.Round(inputs.Length * 1.66f);
			string[] activations = new string[] { "leakyrelu", "leakyrelu", "leakyrelu", "leakyrelu" };
			NeuralNetwork nnData = this.bestNeuralNetwork ?? new NeuralNetwork(this, new int[] { inputs.Length, hiddenSize, hiddenSize, inputs.Length }, activations);
			nnData.DeepCopyLayersInfo(out data.layersInfo);
			nnData.DeepCopyBiases(out data.biases);
			nnData.DeepCopyNeurons(out data.neurons);
			nnData.DeepCopyWeights(out data.weights);
			nnData.DeepCopyActivations(out data.activations);

			var json = ConfigService.SaveWithType(data);

			File.WriteAllText(path, json);
			CurrentFileName = path;
			Settings.RecentBotFiles.Add(CurrentFileName);
			MessageLabel.Text = $"{Path.GetFileName(CurrentFileName)} saved";
		}

		protected override void NewMenuItem_Click(object sender, EventArgs e)
		{
			CurrentFileName = "";
			this.bestNeuralNetwork.BotFitness.is_Reset = true;

			foreach (var cp in ControlProbabilityPanel.Controls.OfType<BotControlsRow>())
			{
				cp.Probability = 0;
			}

			FrameLength = 0;
			MaximizeAddress = 0;
			TieBreaker1Address = 0;
			TieBreaker2Address = 0;
			TieBreaker3Address = 0;
			StartFromSlotBox.SelectedIndex = 0;
			MainOperator.SelectedIndex = 0;
			Tiebreak1Operator.SelectedIndex = 0;
			Tiebreak2Operator.SelectedIndex = 0;
			Tiebreak3Operator.SelectedIndex = 0;
			MainBestRadio.Checked = true;
			MainValueNumeric.Value = 0;
			TieBreak1Numeric.Value = 0;
			TieBreak2Numeric.Value = 0;
			TieBreak3Numeric.Value = 0;
			TieBreak1BestRadio.Checked = true;
			TieBreak2BestRadio.Checked = true;
			TieBreak3BestRadio.Checked = true;

			UpdateBestAttempt();
			UpdateComparisonBotAttempt();
		}

		protected override void ClearBestButton_Click(object sender, EventArgs e)
		{
			this.bestNeuralNetwork.BotFitness.is_Reset = true;
			Attempts = 0;
			Frames = 0;
			UpdateBestAttempt();
			UpdateComparisonBotAttempt();
		}

		protected override void PlayBestButton_Click(object sender, EventArgs e)
		{
			StopBot();
			_replayMode = true;
			_doNotUpdateValues = true;

			// here we need to apply the initial frame's input from the best attempt
			var logEntry = this.bestNeuralNetwork.BotFitness.Log[0];
			var controller = MovieSession.GenerateMovieController();
			controller.SetFromMnemonic(logEntry);
			foreach (var button in controller.Definition.BoolButtons)
			{
				// TODO: make an input adapter specifically for the bot?
				InputManager.ButtonOverrideAdapter.SetButton(button, controller.IsPressed(button));
			}

			InputManager.SyncControls(Emulator, MovieSession, Config);

			MainForm.LoadQuickSave(SelectedSlot, true); // Triggers an UpdateValues call
			_lastFrameAdvanced = Emulator.Frame;
			_doNotUpdateValues = false;
			_startFrame = Emulator.Frame;
			SetNormalSpeed();
			UpdateBotStatusIcon();
			MessageLabel.Text = "Replaying";
			MainForm.UnpauseEmulator();
		}

		protected override void MainBestRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				MainValueNumeric.Enabled = false;
				FitnessComparison.Maximize = this.bestNeuralNetwork.BotFitness?.Maximize ?? 0;
			}
		}

		protected override void Tiebreak1BestRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				TieBreak1Numeric.Enabled = false;
				FitnessComparison.TieBreak1 = this.bestNeuralNetwork.BotFitness?.TieBreak1 ?? 0;
			}
		}

		protected override void Tiebreak2BestRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				TieBreak2Numeric.Enabled = false;
				FitnessComparison.TieBreak2 = this.bestNeuralNetwork.BotFitness?.TieBreak2 ?? 0;
			}
		}

		protected override void Tiebreak3BestRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				TieBreak3Numeric.Enabled = false;
				FitnessComparison.TieBreak3 = this.bestNeuralNetwork.BotFitness?.TieBreak3 ?? 0;
			}
		}

		protected override void MainValueRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				MainValueNumeric.Enabled = true;
				FitnessComparison.Maximize = (int) MainValueNumeric.Value;
			}
		}

		protected override void TieBreak1ValueRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				TieBreak1Numeric.Enabled = true;
				FitnessComparison.TieBreak1 = (int) TieBreak1Numeric.Value;
			}
		}

		protected override void TieBreak2ValueRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				TieBreak2Numeric.Enabled = true;
				FitnessComparison.TieBreak2 = (int) TieBreak2Numeric.Value;
			}
		}

		protected override void TieBreak3ValueRadio_CheckedChanged(object sender, EventArgs e)
		{
			if (sender is RadioButton radioButton && radioButton.Checked)
			{
				TieBreak3Numeric.Enabled = true;
				FitnessComparison.TieBreak3 = (int) TieBreak3Numeric.Value;
			}
		}

		protected override void MainValueNumeric_ValueChanged(object sender, EventArgs e)
		{
			NumericUpDown numericUpDown = (NumericUpDown) sender;
			FitnessComparison.Maximize = (int) numericUpDown.Value;
		}

		protected override void TieBreak1Numeric_ValueChanged(object sender, EventArgs e)
		{
			NumericUpDown numericUpDown = (NumericUpDown) sender;
			FitnessComparison.TieBreak1 = (int) numericUpDown.Value;
		}

		protected override void TieBreak2Numeric_ValueChanged(object sender, EventArgs e)
		{
			NumericUpDown numericUpDown = (NumericUpDown) sender;
			FitnessComparison.TieBreak2 = (int) numericUpDown.Value;
		}

		protected override void TieBreak3Numeric_ValueChanged(object sender, EventArgs e)
		{
			NumericUpDown numericUpDown = (NumericUpDown) sender;
			FitnessComparison.TieBreak3 = (int) numericUpDown.Value;
		}

		public double Evaluate(IChromosome chromosome)
		{
			Gene[] genes = chromosome.GetGenes();
			IList<int> indices = new List<int>();
			int fitness = 1;
			int previousIndex = Convert.ToInt32(genes[0].Value, CultureInfo.InvariantCulture);
			indices.Add(previousIndex);

			foreach (Gene g in genes)
			{
				int currentIndex = Convert.ToInt32(g.Value, CultureInfo.InvariantCulture);

				NeuralNetwork currentNetwork = this.Networks[currentIndex];
				NeuralNetwork previousNetwork = this.Networks[previousIndex];
				fitness += (IsBetter(previousNetwork.BotFitness, currentNetwork.BotFitness) ? 1 : 0);

				previousIndex = currentIndex;
				indices.Add(previousIndex);
			}

			fitness += (IsBetter(Networks[indices.Last()].BotFitness, Networks[indices.First()].BotFitness) ? 1 : 0);
			((NeuralNetwork) chromosome).SetChromosomeFitness(fitness);

			int repeat = this.Networks.Count - indices.Distinct().Count();
			if (repeat > 0)
			{
				fitness /= repeat;
			}

			if (fitness <= 0)
			{
				fitness = 1;
			}

			// Fitness cannot be 0 or less.
			return fitness;
		}
	}

	/*
	Neural Network code starts from here.
	*/
	public class NeuralNetwork : ChromosomeBase
	{
		private int[] layersInfo;
		private int[] activations;
		private float[][] neurons;
		private float[][] biases;
		private float[][][] weights;

		private const float learningRate = 1.0e-6f;
		public float cost = 0f;

		private float[][] deltaBiases;
		private float[][][] deltaWeights;
		private int deltaCount;

		public BotAttempt BotFitness { get; set; }

		public int[] LayersInfo { get { return layersInfo; } set { layersInfo = value; } }
		public float[][] Biases { get { return biases; } set { biases = value; } }
		public float[][] Neurons { get { return neurons; } set { neurons = value; } }
		public float[][][] Weights { get { return weights; } set { weights = value; } }

		private static Random rand = new Random();

		private NeuralNetworkBot Owner { get; set; }

		public NeuralNetwork(NeuralNetworkBot owner, int inputSize, int[] layersInfo, int[] activations) : base(layersInfo.Length)
		{
			this.Owner = owner;
			this.layersInfo = new int[layersInfo.Length];
			Array.Copy(layersInfo, this.layersInfo, layersInfo.Length);
			this.activations = new int[activations.Length];
			Array.Copy(activations, this.activations, activations.Length);
			this.InitializeNeuralNetwork(this.layersInfo, this.activations);
		}

		public NeuralNetwork(NeuralNetworkBot owner, int[] layers, float[][] biases, float[][] neurons, float[][][] weights, int[] activations) : base(layers.Length)
		{
			this.Owner = owner;

			this.layersInfo = (int[]) layers?.Clone();

			this.biases = new float[biases.Length][];
			for (int i = 0; i < biases.Length; i++)
			{
				this.biases[i] = (float[]) biases[i].Clone();
			}

			this.neurons = new float[neurons.Length][];
			for (int i = 0; i < biases.Length; i++)
			{
				this.neurons[i] = (float[]) neurons[i].Clone();
			}

			this.weights = new float[weights.Length][][];
			for (int i = 0; i < weights.Length; i++)
			{
				this.weights[i] = (float[][]) weights[i].Clone();
				for (int j = 0; j < weights[i].Length; j++)
				{
					this.weights[i][j] = (float[]) weights[i][j].Clone();
				}
			}

			this.activations = (int[]) activations.Clone();

			this.BotFitness = new BotAttempt();

			CreateGenes();
		}

		public NeuralNetwork(NeuralNetworkBot owner, int[] layers, string[] layerActivations) : base(layers.Length)
		{
			this.Owner = owner;
			this.activations = new int[layerActivations.Length];
			for (int i = 0; i < layerActivations.Length; i++)
			{
				string action = layerActivations[i];
				switch (action)
				{
					case "sigmoid":
						this.activations[i] = 0;
						break;
					case "tanh":
						this.activations[i] = 1;
						break;
					case "relu":
						this.activations[i] = 2;
						break;
					case "leakyrelu":
						this.activations[i] = 3;
						break;
					default:
						this.activations[i] = 2;
						break;
				}
			}
			InitializeNeuralNetwork(layers, this.activations);
		}

		public NeuralNetwork(NeuralNetworkBot owner, int[] layers, int[] activations) : base(layers.Length)
		{
			this.Owner = owner;
			this.activations = new int[activations.Length];
			Array.Copy(activations, this.activations, activations.Length);
			InitializeNeuralNetwork(layers, this.activations);
		}

		public NeuralNetwork(NeuralNetwork other) : base(other.layersInfo.Length)
		{
			InitializeNeuralNetwork(other.layersInfo, other.activations);
			DeepCopy(other);
		}

		public void InitializeNeuralNetwork(int[] layersInfo, int[] layerActivations)
		{
			// Set layers info
			this.layersInfo = new int[layersInfo.Length];
			Array.Copy(layersInfo, 0, this.layersInfo, 0, layersInfo.Length);

			// Set fitness
			this.BotFitness = new BotAttempt();

			// Generate matrices
			InitializeNeurons();
			InitializeBiases();
			InitializeWeights();
			CreateGenes();
		}

		private void InitializeNeurons()
		{
			List<float[]> neuronsList = new List<float[]>();
			foreach (int layerSize in layersInfo)
			{
				neuronsList.Add(new float[layerSize]);
			}
			neurons = neuronsList.ToArray();
		}

		private void InitializeBiases()
		{
			List<float[]> biasList = new List<float[]>();
			for (int i = 1; i < this.layersInfo.Length; i++)
			{
				float[] bias = new float[this.layersInfo[i]];
				for (int j = 0; j < this.layersInfo[i]; j++)
				{
					bias[j] = ((float) rand.NextDouble() - 0.5f);
				}
				biasList.Add(bias);
			}
			this.biases = biasList.ToArray();
		}

		private void InitializeWeights()
		{
			List<float[][]> weightsList = new List<float[][]>();
			for (int i = 1; i < layersInfo.Length; i++)
			{
				List<float[]> layerWeightsList = new List<float[]>();
				int neuronsInPreviousLayer = layersInfo[i - 1];
				for (int j = 0; j < neurons[i].Length; j++)
				{
					float[] neuronWeights = new float[neuronsInPreviousLayer];
					for (int k = 0; k < neuronsInPreviousLayer; k++)
					{
						// Set the weights to be in the (-0.5f, 0.5f) range.
						neuronWeights[k] = 0f; //(float) (rand.NextDouble() - 0.5f);
					}
					layerWeightsList.Add(neuronWeights);
				}
				weightsList.Add(layerWeightsList.ToArray());
			}
			weights = weightsList.ToArray();
		}

		public NeuralNetwork DeepCopy(NeuralNetwork source)
		{
			// Deep copy biases
			this.Owner = source.Owner;
			source.DeepCopyLayersInfo(out this.layersInfo);
			source.DeepCopyBiases(out this.biases);
			source.DeepCopyNeurons(out this.neurons);
			source.DeepCopyWeights(out this.weights);
			source.DeepCopyActivations(out this.activations);
			this.BotFitness = (source.BotFitness != null ? new BotAttempt(source.BotFitness) : new BotAttempt());
			return this;
		}

		public void DeepCopyLayersInfo(out int[] target)
		{
			target = (int[]) this.layersInfo.Clone();
		}

		public void DeepCopyBiases(out float[][] target)
		{
			target = new float[this.biases.Length][];
			for (int i = 0; i < target.Length; i++)
			{
				target[i] = (float[]) this.biases[i].Clone();
			}
		}

		public void DeepCopyNeurons(out float[][] target)
		{
			target = new float[this.neurons.Length][];
			for (int i = 0; i < target.Length; i++)
			{
				target[i] = (float[]) this.neurons[i].Clone();
			}
		}

		public void DeepCopyWeights(out float[][][] target)
		{
			target = new float[this.weights.Length][][];
			for (int i = 0; i < target.Length; i++)
			{
				target[i] = (float[][]) this.weights[i].Clone();
				for (int j = 0; j < target[i].Length; j++)
				{
					target[i][j] = (float[]) this.weights[i][j].Clone();
				}
			}
		}

		public void DeepCopyActivations(out int[] target)
		{
			target = (int[]) this.activations.Clone();
		}

		public float[] ForwardPropagation(float[] inputs)
		{
			for (int i = 0; i < inputs.Length; i++)
			{
				neurons[0][i] = inputs[i];
			}

			for (int i = 1; i < this.layersInfo.Length; i++)
			{
				for (int j = 0; j < this.layersInfo[i]; j++)
				{
					float value = 0f;
					for (int k = 0; k < this.layersInfo[i - 1]; k++)
					{
						value += weights[i - 1][j][k] * neurons[i - 1][k];
					}
					neurons[i][j] = (float) Activate(value + biases[i - 1][j], i - 1);
				}
			}
			return neurons[neurons.Length - 1];
		}

		public void BackPropagate(float[] inputs, float[] expected)
		{
			float[] output = ForwardPropagation(inputs);

			// Preparing error cost calculation.
			cost = 0f;
			for (int i = 0; i < output.Length; i++)
			{
				cost += (float) Math.Pow(output[i] - expected[i], 2);
			}
			cost /= 2;

			// Gamma initializaiton.
			float[][] gamma;
			List<float[]> gammaList = new List<float[]>();
			for (int i = 0; i < this.layersInfo.Length; i++)
			{
				gammaList.Add(new float[this.layersInfo[i]]);
			}
			gamma = gammaList.ToArray();

			// Gamma calculation
			for (int i = 0; i < output.Length; i++)
			{
				gamma[this.layersInfo.Length - 1][i] = (output[i] - expected[i]) * ActivateDerivative(output[i], this.layersInfo.Length - 2);
			}

			// Weight and bias calculations
			for (int i = 0; i < this.layersInfo[this.layersInfo.Length - 1]; i++)
			{
				this.biases[this.layersInfo.Length - 2][i] -= gamma[this.layersInfo.Length - 1][i] * learningRate;
				for (int j = 0; j < this.layersInfo[this.layersInfo.Length - 2]; j++)
				{
					this.weights[this.layersInfo.Length - 2][i][j] -= gamma[this.layersInfo.Length - 1][i] * this.neurons[this.layersInfo.Length - 2][j] * learningRate;
				}
			}

			// Hidden layers calculations
			// Going backwards
			for (int i = this.layersInfo.Length - 2; i > 0; i--)
			{
				for (int j = 0; j < this.layersInfo[i]; j++)
				{
					gamma[i][j] = 0f;
					for (int k = 0; k < gamma[i + 1].Length; k++)
					{
						gamma[i][j] += gamma[i + 1][k] * this.weights[i][k][j];
					}
					gamma[i][j] *= ActivateDerivative(this.neurons[i][j], i - 1);
				}

				for (int j = 0; j < this.layersInfo[i]; j++)
				{
					this.biases[i - 1][j] -= gamma[i][j] * learningRate;
					for (int k = 0; k < this.layersInfo[i - 1]; k++)
					{
						this.weights[i - 1][j][k] -= gamma[i][j] * this.neurons[i - 1][k] * learningRate;
					}
				}
			}
		}

		public void Mutate(int high, float value)
		{
			for (int i = 0; i < this.biases.Length; i++)
			{
				for (int j = 0; j < this.biases[i].Length; j++)
				{
					this.biases[i][j] = (((float) rand.NextDouble() * high) <= 2f) ? this.biases[i][j] += (((float) rand.NextDouble() * 2f - 1f) * value) : this.biases[i][j];
				}
			}

			for (int i = 0; i < weights.Length; i++)
			{
				for (int j = 0; j < weights[i].Length; j++)
				{
					for (int k = 0; k < weights[i][j].Length; k++)
					{
						weights[i][j][k] = (((float) rand.NextDouble() * high) <= 2f) ? weights[i][j][k] += (((float) rand.NextDouble() * 2f - 1f) * value) : weights[i][j][k];
					}
				}
			}
		}

		public float Activate(float value, int layer)
		{
			switch (this.activations[layer])
			{
				case 0:
					return (float) Sigmoid(value);
				case 1:
					return (float) Tanh(value);
				case 2:
					return (float) ReLU(value);
				case 3:
					return (float) LeakyReLU(value);
				default:
					return (float) ReLU(value);
			}
		}

		public float ActivateDerivative(float value, int layer)
		{
			switch (this.activations[layer])
			{
				case 0:
					return (float) SigmoidDerivative(value);
				case 1:
					return (float) TanhDerivative(value);
				case 2:
					return (float) ReLUDerivative(value);
				case 3:
					return (float) LeakyReLUDerivative(value);
				default:
					return (float) ReLUDerivative(value);
			}
		}

		public static double Sigmoid(double x)
		{
			return 1.0 / (1.0 + Math.Exp(-x));
		}

		public static double LeakyReLU(double x)
		{
			if (x >= 0)
				return x;
			return x / 20.0;
		}

		public static double Tanh(double x)
		{
			return Math.Tanh(x);
		}

		public static double ReLU(double x)
		{
			return (0 >= x) ? 0 : x;
		}

		public static double SigmoidDerivative(double x)
		{
			return x * (1 - x);
		}

		public static double TanhDerivative(double x)
		{
			return 1 - (x * x);
		}

		public static double ReLUDerivative(double x)
		{
			return (0 >= x) ? 0 : 1;
		}

		public static double LeakyReLUDerivative(double x)
		{
			return (0 >= x) ? 0.01f : 1;
		}

		public override Gene GenerateGene(int geneIndex)
		{
			Gene g = new Gene(RandomizationProvider.Current.GetInt(3, NeuralNetworkBot.NeuralNetworkPoolSize - 1));
			return g;
		}

		public override IChromosome CreateNew()
		{
			float[] inputs = Owner.GetCachedInputProbabilities();
			NeuralNetwork result = new NeuralNetwork(this);
			result.BackPropagate(NeuralNetworkBot.FeedForwards, inputs);
			result.Mutate((int) (1.0f / NeuralNetworkBot.MutationChance), NeuralNetworkBot.MutationStrength);
			return result;
		}

		public override IChromosome Clone()
		{
			var copy = base.Clone() as NeuralNetwork;
			copy = copy.DeepCopy(this);
			copy.InitializeNeuralNetwork(copy.layersInfo, copy.activations);
			return copy;
		}

		public void SetChromosomeFitness(int fitness)
		{
			this.BotFitness.Maximize = fitness;
			this.BotFitness.TieBreak1 = (int) Math.Round((fitness * 500f) / this.layersInfo.Length, 0);
			this.BotFitness.TieBreak2 = (int) Math.Round((fitness * 1100f) / this.layersInfo.Length, 0);
			this.BotFitness.TieBreak3 = (int) Math.Round((fitness * 1900f) / this.layersInfo.Length, 0);
		}
	}

	public class NeuralNetworkBotData : BotData
	{
		public int[] layersInfo;
		public float[][] biases;
		public float[][] neurons;
		public float[][][] weights;
		public int[] activations;

		public int[] LayersInfo { get { return layersInfo; } set { layersInfo = value; } }
		public float[][] Biases { get { return biases; } set { biases = value; } }
		public float[][] Neurons { get { return neurons; } set { neurons = value; } }
		public float[][][] Weights { get { return weights; } set { weights = value; } }
		public int[] Activations { get { return activations; } set { activations = value; } }
	}
}

