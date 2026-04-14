function summary = compare_L_prediction_pipeline(msc_ref_mode, snv_mode)
if nargin < 1, msc_ref_mode = []; end
if nargin < 2, snv_mode = []; end
summary = compare_property_prediction_pipeline('L*', msc_ref_mode, snv_mode);
end
