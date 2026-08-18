import logging
import math
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import zarr
from Bio import AlignIO, ExPASy, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from massivora import cpp_bindings
from massivora.utils import get_cuda_module, gpu_is_available

logger = logging.getLogger(__name__)

class TextAlignment(MultipleSeqAlignment):
    def __init__(self, records):
        # Allowing contruction from file path
        if type(records) == str:
            if os.path.isfile(records):
                logger.info(f'Loading alignment from file: {records}')
                _, ext = os.path.splitext(records)
                self.alignmentName = os.path.basename(_)
                if ext.lower() == ".sto":
                    fmt = "stockholm"
                elif ext.lower().startswith(".fa") or ext.lower() == ".a2m":
                    fmt = "fasta"
                else:
                    logger.error(f'Input file {os.path.basename(_)} has an unsupported format {ext}')
                    raise NotImplementedError
                records = AlignIO.read(records, fmt)
            else: raise FileNotFoundError(f"File not found: {records}")
        elif isinstance(records, SeqRecord):
            # construct with single record
            records = [records]
        # construct with biopython supported iterable
        super().__init__(records)
        try:
            self.querySequence = self._records[0]
            self._update_info()
        except IndexError:
            logger.error("Alignment is empty")
            raise ValueError("Alignment is empty")
        self.saved_columns = []
        logger.debug(f'Alignment loaded: {len(self)} sequences, {len(self[0])} positions')

    def _parse_description(self, description):
        # TODO: Try the taxid first, if not exist, try the species name
        pattern = r"OS=(.*?)(?=\s+(?:OX|GN|PE|SV)=|$)"\
                + r"(?:\sOX=(.*?))?(?=\s+(?:GN|PE|SV)=|$)"\
                + r"(?:\sGN=(.*?))?(?=\s+(?:PE|SV)=|$)"\
                + r"(?:\sPE=(.*?))?(?=\s+(?:SV)=|$)"\
                + r"(?:\sSV=(.*?))?$"
        #r"OS=(.*?)(?=\s+(?:GN|PE|SV)=|$)(?:\s(?:GN)=(.*?))?(?=\s+(?:PE|SV)=|$)(?:\s(?:PE)=(.*?))?(?=\s+(?:SV)=|$)(?:\s(?:SV)=(.*?))?$"
        match = re.search(pattern, description)
        if match:
            # matching pattern OS=... OX=... GN=... PE=... SV=...
            return match.group(1), match.group(2), match.group(3), match.group(4), match.group(5)
        return None, None, None, None, None

    def _analyze_descriptions(self):
        self._desc_list = []
        for record in self._records:
            os_, ox, gn, pe, sv = self._parse_description(record.description)
            self._desc_list.append({
                "id": record.id,
                "OS": os_,
                "OX": int(ox) if ox else None,
                "GN": gn,
                "PE": pe,
                "SV": sv
            })

    def _update_info(self):
        self.sequenceLength = len(self._records[0].seq)
        self.sequenceCount = len(self._records)
        self.matrix = self.Get_Numpy_Array()
        self.encoded_matrix = self.Get_Numpy_Array(mapped=True)

    @classmethod
    def From_SwissProt(cls, protein_id):
        """
        Create an instance from SwissProt record. This function will download 
        protein record from the SwissProt server and then load it. 
        
        If the user wish to load an already downloaded file, they should load this file 
        with BioPython, and then use the constructor of this class to create an instance.

        >>> with open("protein.txt") as f:
        >>>     record = SeqIO.read(f, 'swiss')
        >>> alignment = TextAlignment(record)

        Parameters
        ----------
        `protein_id` — str
            Uniprot ID of the protein

        Returns
        -------
        `TextAlignment`
            The alignment instance of that only contains this protein
        """
        try:
            handle = ExPASy.get_sprot_raw(protein_id)
            record = SeqIO.read(handle, 'swiss')
            return cls(record)
        except ValueError as e:
            logger.error(f"Failed to fetch protein {protein_id}: {e}")
            return None

    def AnalogueSearch(self, query=None, threshold=0.2, output_prefix=None, jackhmmer_path=None, iterations=5, threads=4, database=None):
        if not query:
            query = output_prefix + '.fasta'
            SeqIO.write(self._records[0], query, "fasta")
            search_threshold = str(threshold * self.sequenceLength)
        else:
            search_threshold = str(threshold * len(query))
        cmd = [
            jackhmmer_path,
            "-N", str(iterations),
            "-o", '/dev/null',
            "-A", output_prefix + '.sto',
            "--tblout", output_prefix + '.tblout',
            "--domtblout", output_prefix + '.domtblout',
            "--noali",
            "--notextw",
            "-T", search_threshold,
            "--domT", search_threshold,
            "--incT", search_threshold,
            "--incdomT", search_threshold,
            "--cpu", str(threads),
            query, database
        ]
        logger.debug("Running Jackhmmer with the following command line: "+ ' '.join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            logger.error(f"Jackhmmer failed with return code {r.returncode}")
            raise RuntimeError(f"Jackhmmer failed with return code {r.returncode}")
        else:
            self.__init__(output_prefix + '.sto')

    def Get_Numpy_Array(self, mapped=False, string=None):
        """
        Transform the MSA into a numpy array.

        Parameters
        ----------
        `mapped` — bool (optional)
            Whether to map the characters into numbers (default: `False`)
        `string` — str (optional)
            The string that contains all sequences in the MSA, 
            concatenated one after another (default: `None`)

        Returns
        -------
        `numpy.ndarray`
            The transformed MSA as a numpy array
        """
        if not string: string = ''.join([str(record.seq) for record in self])
        matrix = cpp_bindings.getAlignmentInNumpy(string, self.sequenceLength, self.sequenceCount)
        # First change all the '.' to '-'
        matrix[matrix == b'.'] = b'-'
        if mapped:
            # Then map the characters to numbers
            # All the invalid characters are mapped to 21
            mapping = np.full(128, 21, dtype=np.int8)
            valid_chars = 'acdefghiklmnpqrstvwy-ACDEFGHIKLMNPQRSTVWY'
            values = np.arange(-20, 21, dtype=np.int8)
            for char, value in zip(valid_chars, values): mapping[ord(char)] = value
            ascii_codes = np.frombuffer(matrix.tobytes(), dtype=np.int8)
            mapped_flat = mapping[ascii_codes]
            matrix = mapped_flat.reshape(matrix.shape)
        return matrix

    def Filtering_MSA_Invalid(self):
        """
        Filtering out the sequences who has invalid characters in the MSA. 
        Valid characters include 20 natural amino acids and the gap character '-'.
        """
        # Remove sequences with invalid characters from the alignment
        valid_rows = np.where(np.max(self.encoded_matrix, axis=1) <= 20)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), valid_rows)
 
        if not hasattr(self, "_desc_list"):
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
        else:
            for i in sorted(indices_to_delete, reverse=True):
                del self._records[i]
                del self._desc_list[i]
        self._update_info()
        logger.info(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Filtering_MSA_Gap(self, row_keep_ratio=0.5, col_keep_ratio=0.5):
        """
        Filtering out the gaps in the MSA according to the set ratios. 
        This function does: 1. Remove the all-gap columns. 2. Remove sequences 
        whose gap ratio are higher than `row_keep_ratio`. 3. Create masks for 
        columns whose gap ratio are higher than `col_keep_ratio`.

        Parameters
        ----------
        `row_keep_ratio` — float (optional)
            Ratio threshold for keep rows (default: `0.5`)
        `col_keep_ratio` — float (optional)
            Ratio threshold for keep columns (default: `0.5`)
        """
        # Compute sequence identity of every sequence to the query
        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        
        # Remove gaps if the input is a stockholm file
        nongap_columns_to_keep = np.array([char != '-' for char in self._records[0].seq])
        matrix = self.matrix.copy()
        encoded_matrix = self.encoded_matrix.copy()
        encoded_matrix = encoded_matrix[:, nongap_columns_to_keep]

        # Calculate the ratio of '-' in each row and filter out rows with a ratio higher than row_keep_ratio
        logger.debug("Calculate rows to keep")
        row_gap_ratios = np.sum(encoded_matrix == 0, axis=1) / encoded_matrix.shape[1]
        indices_to_keep = np.where(row_gap_ratios <= row_keep_ratio)[0]
        indices_to_delete = np.setdiff1d(np.arange(len(self._records)), indices_to_keep)
        encoded_matrix = encoded_matrix[indices_to_keep]
        logger.debug(f"Matrix shape after row filtering: {encoded_matrix.shape}")

        # Calculate the ratio of '-' in each column and filter out columns with a ratio higher than col_keep_ratio
        logger.debug("Calculate columns to keep")
        L = encoded_matrix.shape[0]
        col_gap_ratios = np.sum(encoded_matrix == 0, axis=0) / L
        cols_to_keep = np.where(col_gap_ratios < col_keep_ratio)[0]
        self.saved_columns = cols_to_keep.tolist() # Just keep it as a mask

        # Calculate identity to query for remaining sequences
        temp_matrix = encoded_matrix.copy()[:, cols_to_keep]
        query_seq = temp_matrix[0]
        ident_to_query = np.round(
            (
                np.sum(temp_matrix == query_seq.reshape(1, -1), axis=1)
                + encoded_matrix.shape[1] # removed columns are treated as matches
                - temp_matrix.shape[1]
            ) / encoded_matrix.shape[1]
        , 3)
        for i, identity in zip(indices_to_keep, ident_to_query):
            self._desc_list[i]["query_identity"] = float(identity)

        logger.debug("Reconstructing alignment")
        matrix = matrix[:, nongap_columns_to_keep]
        matrix = matrix[indices_to_keep]

        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
            del self._desc_list[i]
        assert len(self._records) == L
        for i, record in enumerate(self):
            record.letter_annotations = {}
            byte_t = bytes()
            for j in matrix[i]: byte_t += j
            record.seq = Seq(byte_t.decode())

        self._update_info()

        logger.info(f'Alignment filtered: {len(self)} sequences, {len(self[0])} positions')

    def Best_Reciprocal_Hit(self, paralog_threshold=0.9, allowed_error=0.02):
        """
        Perform the best reciprocal hit scheme for paralog filtering on this 
        alignment instance. 

        Parameters
        ----------
        `paralog_threshold` — float (optional)
            Threshold for similarity check (default: `0.9`)
        `allowed_error` — float (optional)
            Allowed similarity error (default: `0.02`)
        """
        # Compute sequence identity of every sequence to the query
        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        encoded_matrix = self.encoded_matrix.copy()
        if self._desc_list[0].get("query_identity") is None:
            # If the identity to query has not been calculated, calculate it now
            if self.saved_columns:
                # Columns that are masked out are treated as matches
                reduced_matrix = encoded_matrix[:, self.saved_columns]
                query_seq = reduced_matrix[0]
                ident_to_query = np.round(np.sum(reduced_matrix == query_seq.reshape(1, -1), axis=1)
                                          + self.sequenceLength
                                          - len(self.saved_columns), 3) / self.sequenceLength
            else:
                query_seq = encoded_matrix[0]
                ident_to_query = np.round(np.sum(encoded_matrix == query_seq.reshape(1, -1), axis=1), 3) / self.sequenceLength
            for i, identity in enumerate(ident_to_query):
                self._desc_list[i]["query_identity"] = float(identity)

        desc_df = pd.DataFrame(self._desc_list)
        indices_to_keep1 = (
            desc_df
            .sort_values(by="query_identity", ascending=False)   # highest -> lowest
            .groupby("OX", sort=False)
            .head(1)                            # keep first (highest identity) per OX
            .index.to_numpy()                   # these are the row indices in self._records
        )
        logger.debug("This alignment has {} unique species.".format(len(indices_to_keep1)))

        # Get query taxid
        query_name = self[0].name.split('/')[0]
        for i, rec in enumerate(self[1:]):
            if query_name in rec.name:
                paralog_taxid = self._desc_list[i+1]["OX"]
                break
        else:
            logger.warning("No paralog taxid could be identified; skipping paralog filtering.")
            return

        # Collect all paralog records sharing this taxid
        paralog_indices = []
        for i, desc in enumerate(self._desc_list):
            if desc["OX"] == paralog_taxid and self._desc_list[i]["query_identity"] < paralog_threshold:
                paralog_indices.append(i)
        if len(paralog_indices) == 0:
            logger.warning(f"No paralog sequences found for identified taxid {paralog_taxid}; skipping paralog filtering.")
            return

        # Calculate identity of each sequence to the paralog group
        if self.saved_columns:
            reduced_matrix = encoded_matrix[:, self.saved_columns]
            paralog_seqs = reduced_matrix[paralog_indices]
            reduced_matrix = reduced_matrix[indices_to_keep1, :]
            ident_to_paralog = np.round(np.sum(reduced_matrix == paralog_seqs.reshape(paralog_seqs.shape[0], 1, -1), axis=2)
                                        + self.sequenceLength - len(self.saved_columns), 3) / self.sequenceLength
        else:
            paralog_seqs = encoded_matrix[paralog_indices]
            ident_to_paralog = np.round(np.sum(encoded_matrix == paralog_seqs.reshape(paralog_seqs.shape[0], 1, -1), axis=2), 3) / self.sequenceLength

        # Compare similarities, if sequence is more similar to any paralog than the query, remove it
        indices_to_keep2 = [0]  # always keep query
        for i, i_all in enumerate(indices_to_keep1):
            max_paralog_id = np.max(ident_to_paralog[:, i])
            if max_paralog_id < self._desc_list[i_all]["query_identity"] + allowed_error:
                indices_to_keep2.append(i_all)
        
        all_indices = np.arange(1, len(self._records))
        indices_to_delete = np.setdiff1d(all_indices, indices_to_keep2)
        logger.debug("Best reciprocal hits retained. The alignment now has {} sequences.".format(len(indices_to_keep2)))

        for i in sorted(indices_to_delete, reverse=True):
            del self._records[i]
            del self._desc_list[i]

        self._update_info()

    def __add__(self, other):
        """
        Contatenate two MSAs according to the sequence taxids. For paralogous 
        sequences, only the one with the highest similarity to the query will 
        be kept.

        Parameters
        ----------
        `other` — TextAlignment
            The other alignment instance

        Returns
        -------
        `TextAlignment`
            The concatenated alignment instance

        Raises
        ------
        `NotImplementedError`
            If the other operand is not an alignment instance
        """
        if not isinstance(other, MultipleSeqAlignment):
            raise NotImplementedError("Only MultipleSeqAlignment can be concatenated")
        
        # Create species-based dictionary for both alignments
        self_species = {}
        other_species = {}

        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()
        if not hasattr(other, "_desc_list"):
            other._analyze_descriptions()

        for rec, desc in zip(self[1:], self._desc_list[1:]):
            taxid = desc["OX"]
            if taxid and taxid not in self_species:
                self_species[taxid] = []
            self_species[taxid].append((rec, desc))

        for rec, desc in zip(other[1:], other._desc_list[1:]):
            taxid = desc["OX"]
            if taxid and taxid not in other_species:
                other_species[taxid] = []
            other_species[taxid].append((rec, desc))

        logger.debug(f"Found {len(self_species)} species in alignment 1")
        logger.debug(f"Found {len(other_species)} species in alignment 2")

        # Merge records from both alignments
        merged = [SeqRecord(self[0].seq + other[0].seq, id='Query', name='Query', description=self[0].id+'_'+other[0].id)]
        common_species = set(self_species.keys()) & set(other_species.keys())
        logger.debug(f"Found {len(common_species)} common species")
        redundant_species = []
        ## First concatenate the sequences that are unique to each alignment
        logger.debug("Looping through common species")
        for species in common_species:
            if len(self_species[species]) == 1 and len(other_species[species]) == 1:
                rec1 = self_species[species][0][0]
                rec2 = other_species[species][0][0]
                # logger.debug(f'rec1.name: {rec1.name}, rec2.name: {rec2.name}, species: {species}')
                new_rec = SeqRecord(rec1.seq + rec2.seq, id=str(species), name=str(species), description=rec1.id+'_'+rec2.id)
                merged.append(new_rec)
            else:
                redundant_species.append(species)
        ## Then concatenate the sequences that are redundant in both alignments
        logger.debug(f"Found {len(redundant_species)} redundant species")
        for species in redundant_species:
            identity = 0
            for rec, desc in self_species[species]:
                if desc["query_identity"] > identity:
                    identity = desc["query_identity"]
                    most_similar_seq_in_self = rec
            identity = 0
            for rec, desc in other_species[species]:
                if desc["query_identity"] > identity:
                    identity = desc["query_identity"]
                    most_similar_seq_in_other = rec
            new_rec = SeqRecord(most_similar_seq_in_self.seq + most_similar_seq_in_other.seq, id=str(species), name=str(species), description=most_similar_seq_in_self.id+'_'+most_similar_seq_in_other.id)
            merged.append(new_rec)
        logger.info(f'Alignment merged: {len(merged)} sequences, {len(merged[0])} positions')
        return TextAlignment(merged)

    def Downsample_Randomly(self, to):
        """
        Downsample the alignment randomly to a lower size

        Parameters
        ----------
        `to` — int
            The number of sequences to downsample to
        """
        if to >= self.sequenceCount:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.sequenceCount} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.sequenceCount), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)
        chosen_records = [self._records[i] for i in chosen_indices]
        super().__init__(chosen_records)
        self._update_info()

    def To_Zarr(self, filename=None, overwrite=False):
        """
        Save the alignment as a Zarr format. The sequences will be grouped by 
        species, and the one with the highest identity to the query will be 
        put at the first position for each species. The species list and saved
        columns will be saved as Zarr attributes. This function will also create
        a species_index_map indicating the starting index of a species.

        Parameters
        ----------
        `filename` — str (optional)
            Filename of the Zarr archive. If not set, 
            will use alignmentName as default (default: `None`)
        `overwrite` — bool (optional)
            Whether to overwrite the existing file (default: `False`)
        """
        if not filename: filename = f'{self.alignmentName}'
        if os.path.exists(os.path.join(filename, 'align', 'zarr.json')):
            if overwrite:
                logger.warning(f"File {filename} already exists. Overwriting it.")
                shutil.rmtree(os.path.join(filename, 'align'), ignore_errors=True)
            else:
                logger.error(f"File {filename} already exists. Use overwrite=True to overwrite it.")
                return
        species_list = ['Query']
        species_index_map = [0]

        if not hasattr(self, "_desc_list"):
            self._analyze_descriptions()

        # Group sequences by species
        species = {}
        for rec, desc in zip(self[1:], self._desc_list[1:]):
            taxid = desc["OX"]
            if taxid is None: continue
            if taxid not in species.keys(): species[taxid] = []
            species[taxid].append(str(rec.seq))

        p_seq = 1
        tempstr = str(self[0].seq)
        for species, sequences in species.items():
            sequence_count = len(sequences)
            # for multiple records of the same species,
            # put the one with the highest identity to the query sequence at the first position
            if sequence_count != 1:
                identity = 0
                top_identity_index = 0
                q_arr = np.array(list(self.querySequence.seq))
                for this_index, rec in enumerate(sequences):
                    r_arr = np.array(list(rec))
                    this_identity = np.sum(q_arr == r_arr)
                    if this_identity > identity:
                        identity = this_identity
                        top_identity_index = this_index
                sequences.insert(0, sequences.pop(top_identity_index))
            for seq in sequences: tempstr += seq
            species_index_map.append(p_seq)
            species_list.append(species)
            p_seq += sequence_count
        matrix = self.Get_Numpy_Array(mapped=True, string=tempstr)
        grp = zarr.open_group(store=filename)
        z = grp.create_array(name='align', shape=(self.sequenceCount, self.sequenceLength), dtype='int8')
        z.attrs.update({
            'species_list': species_list, 
            'species_index_map': species_index_map,
            'saved_columns': [list(self.saved_columns)],
            'lengthes': [len(self[0].seq)]
        })
        z[:] = matrix


class BinaryAlignment(object):
    def __init__(self, filename=None):
        self.matrix = np.zeros((0, 0), dtype=np.int8)
        self.species_list_map = {}
        self.saved_columns = []
        self.lengthes = []

        if isinstance(filename, str):
            try:
                arr = zarr.open_group(filename, mode='r')
                align = arr['align']
                self.matrix = align[:]
                self.weights = align.attrs.get('weights', None)
                self.Beff = align.attrs.get('Beff', None)
                self.saved_columns = align.attrs.get('saved_columns', [])
                self.lengthes = align.attrs.get('lengthes', [self.matrix.shape[1]])
                self.species_list = align.attrs.get('species_list', [])
                self.species_index_map = align.attrs.get('species_index_map', {})
                self.species_list_map = {species: i for i, species in enumerate(self.species_list)}
            except Exception as e:
                raise FileNotFoundError(f"Error loading Zarr file {filename}")

    def __repr__(self):
        return "<%s instance (%i records of length %i) at %x>" % (
            self.__class__,
            self.matrix.shape[0],
            self.matrix.shape[1],
            id(self),
        )

    def __str__(self):
        return f"Alignment from Zarr with {len(self.species_list)} species"

    def __add__(self, other):
        """
        Concatenate the two MSAs according to the sequence taxids

        Parameters
        ----------
        `other` — BinaryAlignment
            The incoming alignment instance

        Returns
        -------
        `BinaryAlignment`
            The concatenated alignment instance
        """
        # Ensure that Query is the first species in both alignments
        common_species = sorted(list(set(self.species_list[1:]) & set(other.species_list[1:])))
        common_species.insert(0, self.species_list[0])

        new_instance = BinaryAlignment()
        new_instance.species_list = common_species
        new_instance.species_index_map = list(range(len(common_species)))
        new_instance.saved_columns = self.saved_columns + other.saved_columns
        new_instance.lengthes = self.lengthes + other.lengthes
        new_instance.species_list_map = {species: i for i, species in enumerate(new_instance.species_list)}

        # species_list_map[species] is the index of the species in the species_list
        # species_index_map[species_list_map[species]] is the index of the species in the matrix
        indices1 = np.array([self.species_index_map[self.species_list_map[species]] for species in common_species])
        indices2 = np.array([other.species_index_map[other.species_list_map[species]] for species in common_species])
        new_instance.matrix = np.concatenate((self.matrix[indices1], other.matrix[indices2]), axis=1)

        return new_instance

    def Gap_Columns_Control(self, gap_ratio=0.5):
        """
        Before this call, saved_columns is a list of per-monomer masks.
        After this call, it becomes a list of per-monomer per-column index maps
        that reflect the gap ratio control. The alignment matrix is not modified.
        Any existing mask is ignored and recomputed from scratch.
        If available, monomer boundaries are derived from lengthes.

        The gap ratio is calculated as the number of gaps in a column divided
        by the total number of sequences. Columns with a gap ratio higher than
        the specified threshold will be excluded from the saved_columns mask.

        Parameters
        ----------
        `gap_ratio` — float (optional)
            Gap ratio in columns (default: `0.5`)
        """
        L, n_cols = self.matrix.shape

        # Ignore any prior mask: all columns are eligible
        if self.lengthes:
            total_len = int(np.sum(self.lengthes))
            if total_len == n_cols:
                block_offsets = [0]
                cursor = 0
                for block_len in self.lengthes[:-1]:
                    cursor += int(block_len)
                    block_offsets.append(cursor)
            else:
                logger.warning(
                    "lengthes sum (%d) does not match alignment width (%d); treating as single monomer",
                    total_len,
                    n_cols,
                )
                block_offsets = [0]
        else:
            # No per-monomer lengths available: treat as single monomer
            block_offsets = [0]

        col_gap_ratios = np.sum(self.matrix == 0, axis=0) / L
        cols_to_keep = np.where(col_gap_ratios < gap_ratio)[0]

        # Rebuild saved_columns as per-monomer per-column maps
        block_ends = block_offsets[1:] + [n_cols]
        new_saved_columns = [[] for _ in range(len(block_offsets))]
        for col in cols_to_keep:
            for block_idx, (start, end) in enumerate(zip(block_offsets, block_ends)):
                if start <= col < end:
                    new_saved_columns[block_idx].append(int(col - start))
                    break

        self.saved_columns = new_saved_columns

    def Downsample_Randomly(self, to):
        if to >= self.matrix.shape[0]:
            logger.error(f"Downsampling to {to} is not possible, as the alignment has only {self.matrix.shape[0]} sequences")
            return
        chosen_indices = np.random.choice(range(1, self.matrix.shape[0]), to-1, replace=False)
        chosen_indices = np.insert(chosen_indices, 0, 0)  # Ensure the first row is included
        self.matrix = self.matrix[chosen_indices]

        # Update metadata to reflect the downsampled alignment
        kept_count = len(chosen_indices)
        new_species_list = ['Query']
        new_species_index_map = list(range(kept_count))

        for i in range(kept_count-1):
            new_species_list.append(self.species_list[chosen_indices[i + 1]])

        self.species_list = new_species_list
        self.species_index_map = new_species_index_map
        self.species_list_map = {species: i for i, species in enumerate(new_species_list)}

    def Reweight_Sequence(self, x=0.8, use_GPU=False, n_threads=0):
        B, N = self.matrix.shape
        identical_threshold = x * N

        if use_GPU:
            if not gpu_is_available():
                logger.error("CuPy is not available. Please install CuPy to use GPU acceleration.")
                raise ImportError("CuPy is not available. Please install CuPy to use GPU acceleration.")
            import cupy as cp

            # The kernel compares four residues at a time
            Nw = (N + 3) // 4
            padN = Nw * 4
            MSA = cp.zeros((B, padN), dtype=cp.int8)
            MSA[:, :N] = cp.asarray(self.matrix, dtype=cp.int8)
            simM = cp.ones(B, dtype=cp.int32)   # every sequence matches itself

            TILE, TY = 32, 8                    # must match RW_TILE / RW_TY
            ntile = math.ceil(B / TILE)
            cuda_module = get_cuda_module()
            pairwise_similarity = cuda_module.get_function('pairwise_similarity_tiled')
            pairwise_similarity(
                (ntile, ntile), (TILE, TY),
                (MSA.view(cp.uint32), simM, np.int32(B), np.int32(Nw),
                 np.int32(padN - N), np.float32(identical_threshold)))
            cp.cuda.runtime.deviceSynchronize()

            m = cp.asnumpy(simM).astype(np.int64)
            del MSA, simM
        else:
            m = np.asarray(
                cpp_bindings.reweightNeighbourCounts(
                    np.ascontiguousarray(self.matrix, dtype=np.int8),
                    identical_threshold, int(n_threads)),
                dtype=np.int64)

        w = 1.0 / m
        self.weights = w.tolist()
        self.Beff = float(w.sum())

    def To_Text(self, restore_gaps=True, filename=None):
        # Create reverse mapping from numbers back to characters
        valid_chars = 'acdefghiklmnpqrstvwy-ACDEFGHIKLMNPQRSTVWY'
        values = np.arange(-20, 21, dtype=np.int8)
        reverse_mapping = np.full(128, ord('X'), dtype=np.uint8)
        for char, value in zip(valid_chars, values):
            reverse_mapping[value + 20] = ord(char)

        # Convert numeric matrix back to character matrix
        ascii_codes = reverse_mapping[self.matrix + 20]
        char_matrix = np.frombuffer(ascii_codes.tobytes(), dtype='S1').reshape(self.matrix.shape)
        if restore_gaps:
            monomer_matrices = []
            cursor = 0
            for monomer_idx, monomer_saved_cols in enumerate(self.saved_columns):
                n = len(monomer_saved_cols)
                monomer_char_matrix = char_matrix[:, cursor:cursor + n]
                cursor += n

                if self.lengthes and monomer_idx < len(self.lengthes):
                    monomer_original_length = int(self.lengthes[monomer_idx])
                elif monomer_saved_cols:
                    monomer_original_length = max(monomer_saved_cols) + 1
                else:
                    monomer_original_length = 0

                monomer_original_matrix = np.full((self.matrix.shape[0], monomer_original_length), b'-', dtype='S1')
                if monomer_saved_cols:
                    monomer_original_matrix[:, monomer_saved_cols] = monomer_char_matrix
                monomer_matrices.append(monomer_original_matrix)

            original_matrix = np.concatenate(monomer_matrices, axis=1)
        else:
            original_matrix = char_matrix.copy()
        original_matrix = original_matrix.astype('U1')

        # Convert matrix back to SeqRecord objects
        seq_records = []
        i = 0
        for species, starting_idx in zip(self.species_list[:-1], self.species_index_map[1:]):
            for _ in range(i, starting_idx):
                row = original_matrix[i]
                sequence_str = ''.join(row.tolist())
                seq_record = SeqRecord(
                    Seq(sequence_str),
                    id=str(i),
                    name=str(i) + ' OX=' + str(species),
                    description='OX=' + str(species)
                )
                seq_records.append(seq_record)
                i += 1
        species = self.species_list[-1]
        for _ in range(i, original_matrix.shape[0]):
            row = original_matrix[i]
            sequence_str = ''.join(row.tolist())
            seq_record = SeqRecord(
                Seq(sequence_str),
                id=str(i),
                name=str(i) + ' OX=' + str(species),
                description='OX=' + str(species)
            )
            seq_records.append(seq_record)
            i += 1

        if filename:
            SeqIO.write(seq_records, filename, "fasta")

        return TextAlignment(seq_records)

    def To_Zarr(self, filename=None, overwrite=False):
        if not filename: filename = 'merged_alignment'
        if os.path.exists(os.path.join(filename, 'align', 'zarr.json')):
            if overwrite:
                logger.warning(f"File {filename} already exists. Overwriting it.")
                shutil.rmtree(os.path.join(filename, 'align'), ignore_errors=True)
            else:
                logger.error(f"File {filename} already exists. Use overwrite=True to overwrite it.")
                return
        grp = zarr.open_group(store=filename)
        z = grp.create_array(name='align', shape=self.matrix.shape, dtype='int8')
        z.attrs.update({
            'saved_columns': getattr(self, 'saved_columns', None),
            'species_list': getattr(self, 'species_list', None),
            'species_index_map': getattr(self, 'species_index_map', None),
            'lengthes': getattr(self, 'lengthes', [self.matrix.shape[1]]),
            'weights': getattr(self, 'weights', None),
            'Beff': getattr(self, 'Beff', None)
        })

        z[:] = self.matrix

    def To_Npy(self, filename=None):
        if filename:
            np.save(filename, self.matrix)
        return self.matrix




