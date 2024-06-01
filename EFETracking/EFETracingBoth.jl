using Plots
using DataFrames

include("EFETrackingC.jl")
include("EFETrackingD.jl")


plot_data_C
plot_data_D

sort!(plot_data_C, [:lr_pB, :time_step])
sort!(plot_data_D, [:lr_pB, :time_step])


plot_data_both = DataFrame(
    lr_pB = plot_data_C.lr_pB,
    time_step = plot_data_C.time_step,
    EFE_C = plot_data_C.average_EFE,
    EFE_D = plot_data_D.average_EFE
)

plot_data_both


res = 0.0:8:1000

filtered_plot_both = filter(row -> row.time_step in res, plot_data_both)

plt = plot(title="Average of EFE for Different lr_pB",
           xlabel="Time Step",
           ylabel="Average EFE difference")

for lr_pB in unique(filtered_plot_both.lr_pB)
    lr_pB_data = filtered_plot_both[filtered_plot_both.lr_pB .== lr_pB, :]
    plot!(plt, lr_pB_data.time_step, lr_pB_data.EFE_C, label="EFE_C lr_pB = $lr_pB")
    plot!(plt, lr_pB_data.time_step, lr_pB_data.EFE_D, label="EFE_D lr_pB = $lr_pB", linestyle=:dash)
end
theme(:juno)

display(plt)
