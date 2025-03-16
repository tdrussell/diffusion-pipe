from deepspeed.pipe import PipelineModule

# PipelineModule partition_method doesn't support uneven partitioning
# This allow for loading more layers into selected GPU
# For example if you have 2 gpus - one with 16GB and other with 24GB normal partitioning would throw OOM
# With this implementation you can set partition_split in config so that less layers is loaded onto 16GB GPU
class ManualPipelineModule(PipelineModule):
    def __init__(self, *args, manual_partition_split=None, **kwargs):
        self.manual_partition_split = manual_partition_split
        super().__init__(*args, **kwargs)

    def _partition_layers(self, method='uniform'):
        if method.lower() == 'manual' and self.manual_partition_split is not None:
            total_layers = len(self._layer_specs)
            boundaries = [0] + self.manual_partition_split + [total_layers]
            stage_id = self._topo.get_coord(self.global_rank).pipe
            self.parts = boundaries
            self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id+1])
        else:
            super()._partition_layers(method)