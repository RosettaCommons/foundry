# MPNN outputs

By default, in the output directory there will be a FASTA-formatted file with the sequence outputs,
as well as a backbone-only CIF-formatted output structure of the designs. 
(The sidechain packing model of LigandMPNN is not currently implemented.)

## Controlling Output

To specify the output directory (required):

    --out_directory DIR/

To turn off FASTA or structure output:

    --write_structures False 
    --write_fasta False

To change the label on the output files (default is to base it on the input filename):

    --name NAME

### Example FASTA Headers

* ProteinMPNN & SolubleMPNN:

    >input_b0_d0, confidence=0.3901, sequence_recovery=0.3823

* LigandMPNN:

    >ligand_b0_d0, confidence=0.4532, ligand_interface_confidence=0.5316, sequence_recovery=0.4187, ligand_interface_sequence_recovery=0.5000

### CIF Output

In addition to the (backbone only) coordinates of the design, the CIF file features `_mpnn_input` and `_mpnn_output` tables recording the parameters of the design. These can be extracted with standard CIF file readers, or through text file processing techniques (e.g. `grep`). 

Standard entries for `_mpnn_output`:

    _mpnn_output.batch_idx                            
    _mpnn_output.design_idx                           
    _mpnn_output.designed_sequence                    
    _mpnn_output.confidence                           
    _mpnn_output.ligand_interface_confidence          
    _mpnn_output.sequence_recovery                    
    _mpnn_output.ligand_interface_sequence_recovery   
    _mpnn_output.model_type                           
    _mpnn_output.checkpoint_path                      
    _mpnn_output.is_legacy_weights                    

Additionally, the `_atom_site` table will be annotated with `_atom_site.mpnn_temperature` and `_atom_site.mpnn_confidence` records giving the per-residue temperature and confidence for the run.
