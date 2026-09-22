from inventory_mdp.cli import main


def test_cli_prints_both_policies(capsys):
    main()
    output = capsys.readouterr().out
    assert "value_iteration / policy_iteration" in output
    assert "0 ->" in output
