using DataFrames
using JLD2
using Plots




file_path = raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\SocCult\ImportantData\DataFiles\results.jld2" 
data = JLD2.load(file_path)

results = data["results"]

results_df = DataFrame(results)

results_df[!, :total_reward] = results_df[!, :score_AIF_agent] .+ results_df[!, :score_AlgoAgent]

x = unique(results_df[!, :lr_pB])
y = unique(results_df[!, :beta])

pivot_df = unstack(results_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

TotalPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Accumulated Total Reward", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg


#### For TitForTat ####
TFT_df = filter(row -> row.AlgoAgent_name == "TitForTatAgent", results_df)

x = unique(TFT_df[!, :lr_pB])
y = unique(TFT_df[!, :beta])

pivot_df = unstack(TFT_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

TFTPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Total Reward against TitForTat", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg

#### For TitFor2Tats ####
TF2T_df = filter(row -> row.AlgoAgent_name == "TitFor2TatsAgent", results_df)

x = unique(TF2T_df[!, :lr_pB])
y = unique(TF2T_df[!, :beta])

pivot_df = unstack(TF2T_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

TF2TPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Total Reward against TitFor2Tats", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg

#### For TwoTitsFor1Tat ####
TTF1T_df = filter(row -> row.AlgoAgent_name == "TwoTitsFor1TatAgent", results_df)

x = unique(TTF1T_df[!, :lr_pB])
y = unique(TTF1T_df[!, :beta])

pivot_df = unstack(TTF1T_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

TTF1TPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Total Reward against TwoTitsFor1Tat", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg

#### For NastyForgivingTFT ####
NFTFT_df = filter(row -> row.AlgoAgent_name == "NastyForgivingTFTAgent", results_df)

x = unique(NFTFT_df[!, :lr_pB])
y = unique(NFTFT_df[!, :beta])

pivot_df = unstack(NFTFT_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

NFTFTPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Total Reward against NastyForgivingTFT", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg

#### For Pavlovian ####
Pav_df = filter(row -> row.AlgoAgent_name == "PavlovianAgent", results_df)

x = unique(Pav_df[!, :lr_pB])
y = unique(Pav_df[!, :beta])

pivot_df = unstack(Pav_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

PavPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Total Reward against Pavlovian", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg

#### For Grofman ####
Grof_df = filter(row -> row.AlgoAgent_name == "GrofmanAgent", results_df)

x = unique(Grof_df[!, :lr_pB])
y = unique(Grof_df[!, :beta])

pivot_df = unstack(Grof_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

GrofPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Total Reward against Grofman", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg

#### For GrimTrigger ####
GT_df = filter(row -> row.AlgoAgent_name == "GrimTriggerAgent", results_df)

x = unique(GT_df[!, :lr_pB])
y = unique(GT_df[!, :beta])

pivot_df = unstack(GT_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

GTPlot = heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Total Reward against GrimTrigger", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
gg

# Combine all the plots
CombinedAlgoPlot = plot(TotalPlot, TFTPlot, TF2TPlot, TTF1TPlot, NFTFTPlot, PavPlot, GrofPlot, GTPlot, layout = (4, 2), size = (1000, 1500))
#savefig(CombinedAlgoPlot, "TotalRewardPlots.svg")

CombinedPaperPlot = plot(TotalPlot, TFTPlot, PavPlot, GrofPlot)
#savefig(CombinedPaperPlot, "PaperPlotsP1.svg")