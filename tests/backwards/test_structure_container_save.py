from pyiron_atomistics import Project, __version__


pr = Project("tests/static/backwards/V{}".format(__version__).replace(".", "_"))
structure = pr.create.structure.ase.bulk("Al")
job = pr.create_job(pr.job_type.StructureContainer, "structure_container")
job.structure = structure
job.save()
